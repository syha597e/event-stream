import jax
import jax.numpy as np
from flax import linen as nn
from .layers import SequenceLayer
from typing import Callable
import numpy


class StackedEncoderModel(nn.Module):
    """ Defines a stack of S5 layers to be used as an encoder.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                     we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    discretization: str
    discretization_first_layer: str
    d_model: int
    d_ssm: int
    block_size: int
    n_layers: int
    num_embeddings: int = 0
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_every_n_layers: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        assert self.num_embeddings > 0
        self.encoder = nn.Embed(num_embeddings=self.num_embeddings, features=self.d_model)

        # generate strides for the model
        layers = []
        d_model = self.d_model
        d_ssm = self.d_ssm
        total_downsampling = 1
        for l in range(self.n_layers):
            # pool from the first layer but don't expand the state dim for the first layer
            stride = self.pooling_stride if l % self.pooling_every_n_layers == 0 else 1
            total_downsampling *= stride
            d_model_in = d_model
            d_model_out = d_model

            if l > 0 and l % self.pooling_every_n_layers == 0:
                d_ssm = self.state_expansion_factor * d_ssm
                d_model_out = self.state_expansion_factor * d_model
                d_model = self.state_expansion_factor * d_model

            layers.append(
                SequenceLayer(
                    ssm=self.ssm,
                    discretization=self.discretization_first_layer if l == 0 else self.discretization,
                    dropout=self.dropout,
                    d_model_in=d_model_in,
                    d_model_out=d_model_out,
                    d_ssm=d_ssm,
                    block_size=self.block_size,
                    activation=self.activation,
                    training=self.training,
                    prenorm=self.prenorm,
                    batchnorm=self.batchnorm,
                    bn_momentum=self.bn_momentum,
                    step_rescale=self.step_rescale,
                    pooling_stride=stride,
                    pooling_mode=self.pooling_mode
                )
            )
        self.layers = layers
        self.total_downsampling = total_downsampling

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        x = self.encoder(x)
        for i, layer in enumerate(self.layers):
            # apply layer SSM
            x, integration_timesteps = layer(x, integration_timesteps)
        return x, integration_timesteps


def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = np.arange(L) < lengths
    return np.sum(mask[..., None]*x, axis=0)/lengths


def timepool(x, integration_timesteps):
    """
    Helper function to perform weighted mean across the sequence length.
    Means are weighted with the integration time steps
    Args:
         x (float32): input sequence (L, d_model)
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    T = np.sum(integration_timesteps, axis=0)
    integral = np.sum(x * integration_timesteps[..., None], axis=0)
    return integral / T


def masked_timepool(x, lengths, integration_timesteps, eps=1e-6):
    """
    Helper function to perform weighted mean across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length. Means are weighted with the integration time steps
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = np.arange(L) < lengths
    T = np.sum(integration_timesteps)

    # integrate with time weighting
    weight = integration_timesteps[..., None] + eps
    integral = np.sum(mask[..., None] * x * weight, axis=0)
    return integral / T


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


class ClassificationModel(nn.Module):
    """ S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            mode        (str):      Options: [pool: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    discretization: str
    discretization_first_layer: str
    d_output: int
    d_model: int
    d_ssm: int
    block_size: int
    n_layers: int
    num_embeddings: int = 0
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_every_n_layers: int = 1
    pooling_mode: str = "last"
    state_expansion_factor: int = 1

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
            ssm=self.ssm,
            discretization=self.discretization,
            discretization_first_layer=self.discretization_first_layer,
            d_model=self.d_model,
            d_ssm=self.d_ssm,
            block_size=self.block_size,
            n_layers=self.n_layers,
            num_embeddings=self.num_embeddings,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            pooling_stride=self.pooling_stride,
            pooling_every_n_layers=self.pooling_every_n_layers,
            pooling_mode=self.pooling_mode,
            state_expansion_factor=self.state_expansion_factor
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        x, length = x  # input consists of data and prepadded seq lens
        # if the sequence is downsampled we need to adjust the length
        length = length // self.encoder.total_downsampling

        x, integration_timesteps = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            x = masked_meanpool(x, length)

        elif self.mode in ["timepool"]:
            # Perform mean pooling across time weighted by integration time steps
            x = masked_timepool(x, length, integration_timesteps)

        elif self.mode in ["last"]:
            # Just take the last state
            x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')
