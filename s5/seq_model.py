import jax
import jax.numpy as np
from flax import linen as nn
from .layers import SequenceLayer
from typing import Callable
from functools import partial
from jax_resnet import pretrained_resnet, slice_variables, Sequential
from flax.core import FrozenDict, frozen_dict


def prep_image_for_transferlearning(image, pad_size):
    """
    Pad a single image with zeros using np.pad.

    Parameters:
    - image: numpy array of shape (height, width, channels)
    - pad_size: tuple (pad_height, pad_width) for padding size

    Returns:
    - padded_image: numpy array of shape (padded_height, padded_width, channels)
    """

    # Pad dimensions for each axis
    pad_width = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (0, 0))

    # Pad image
    padded_image = np.pad(image, pad_width, mode="constant")
    padded_image = np.repeat(padded_image[:, :, np.newaxis, :], 3, axis=2)
    return padded_image.reshape(1, 224, 224, 3)


def merge_events(data, method="mean", flatten=False):
    if method == "mean":
        data = jax.numpy.mean(data, axis=2)
    else:
        assert NotImplementedError("choose mean")

    if flatten:
        return data.flatten()
    else:
        return data


class CNNModule(nn.Module):
    @nn.compact
    def __call__(self, x, pretrained=False):
        if pretrained:
            x = prep_image_for_transferlearning(x, (48, 48))
            x = x.reshape(1, 224, 224, 3)
            # model, variables = get_model_and_variables('resnet50', 0)
            # out,batch_stats = model.apply({'params': variables['params'],'batch_stats': variables['batch_stats']},x, train=True,rngs={'dropout': jax.random.PRNGKey(0)}, mutable='batch_stats')
            resnet_tmpl, params = pretrained_resnet(50)
            resnet_head = resnet_tmpl()
            start, end = 0, len(resnet_head.layers) - 1
            backbone = Sequential(resnet_head.layers[start:end])
            backbone_params = slice_variables(params, start, end)
            out = backbone.apply(backbone_params, x, mutable=False)
            # out = backbone(x,training=False)
            return out.reshape(
                -1,
            )

        else:
            x = nn.Conv(features=64, kernel_size=(11, 11), strides=4, padding=2)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
            x = nn.Conv(features=192, kernel_size=(5, 5), padding=2)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
            x = nn.Conv(features=384, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
            x = nn.Conv(features=256, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
            x = nn.Conv(features=256, kernel_size=(3, 3), padding=1)(x)
            x = nn.relu(x)
            x = x.flatten()
            x = nn.Dense(512)(x)
            return x


class StackedEncoderModel(nn.Module):
    """Defines a stack of S5 layers to be used as an encoder.
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
    n_layers: int
    tokenized: bool = False
    num_embeddings: int = 0
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    use_cnn: bool = True
    use_pretrained: bool = False

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        # don't use bias to void zero tokens to produce an input
        if self.use_cnn:
            self.conv_layer = CNNModule()
        if self.tokenized:
            assert self.num_embeddings > 0
            print("Using tokenized input")
            self.encoder = nn.Embed(
                num_embeddings=self.num_embeddings, features=self.d_model
            )
        else:
            self.encoder = nn.Dense(self.d_model)

        self.layers = [
            SequenceLayer(
                ssm=self.ssm,
                discretization=(
                    self.discretization_first_layer if l == 0 else self.discretization
                ),
                dropout=self.dropout,
                d_model=self.d_model,
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
            )
            for l in range(self.n_layers)
        ]

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        x = x.reshape(67, 128, 128, 2)  # FIXME - hardcode
        if self.use_cnn:
            flat_merge_events = partial(merge_events, flatten=False)
            x = jax.vmap(flat_merge_events)(x)
            x = x.reshape(67, 128, 128, 1)  # FIXME hardcode
            if (
                self.use_pretrained
            ):  # TODO - vmap giving memory error for pretrained model
                # batch_pad = partial(prep_image_for_transferlearning,pad_size=(48,48))
                output_inner = []
                for i in range(67):
                    output_inner.append(self.conv_layer(x[i], pretrained=True))
                cur_layer_input = np.stack(output_inner)
                # x = jax.vmap(batch_pad)(x)
                # batch_conv_layer = partial(self.conv_layer,pretrained=True)
                # cur_layer_input = jax.vmap(batch_conv_layer)(x)
            else:
                cur_layer_input = jax.vmap(self.conv_layer)(x)

        else:
            x = jax.vmap(merge_events)(x)
            x = x.reshape(67, 128, 128, 1)  # FIXME hardcode
            cur_layer_input = x
        x = self.encoder(cur_layer_input)
        for layer in self.layers:
            x = layer(x, integration_timesteps)
        return x


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
    return np.sum(mask[..., None] * x, axis=0) / lengths


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
    # using the Trapezoidal rule to integrate
    integral = np.sum((x[1:] + x[:-1]) / 2 * integration_timesteps[..., None], axis=0)
    return integral / T


def masked_timepool(x, lengths, integration_timesteps):
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
    mask = np.arange(L - 1) < lengths
    T = np.sum(integration_timesteps)
    # using the Trapezoidal rule to integrate
    integral = np.sum(
        mask[..., None] * (x[1:] + x[:-1]) / 2 * integration_timesteps[..., None],
        axis=0,
    )
    return integral / T


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


class ClassificationModel(nn.Module):
    """S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
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
            use_cnn (bool): apply CNN layers to the input frames
            use_pretrained (bool): apply trained resnet50 layers to the input frames instead of custom CNN layer if use_cnn=True
    """

    ssm: nn.Module
    discretization: str
    discretization_first_layer: str
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    tokenized: bool = False
    num_embeddings: int = 0
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    use_cnn: bool = False
    use_pretrained: bool = False

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
            ssm=self.ssm,
            discretization=self.discretization,
            discretization_first_layer=self.discretization_first_layer,
            d_model=self.d_model,
            n_layers=self.n_layers,
            tokenized=self.tokenized,
            num_embeddings=self.num_embeddings,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_cnn=self.use_cnn,
            use_pretrained=self.use_pretrained,
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
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens
        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = np.mean(x, axis=0)
        elif self.mode in ["timepool"]:
            if self.padded:
                x = masked_timepool(x, length, integration_timesteps)
            else:
                x = timepool(x, integration_timesteps)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError(
                    "Mode must be in ['pool'] for self.padded=True (for now...)"
                )
            else:
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
    variable_axes={
        "params": None,
        "dropout": None,
        "batch_stats": None,
        "cache": 0,
        "prime": None,
    },
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)


# For Document matching task (e.g. AAN)
class RetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
    """

    d_model: int
    d_output: int

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        self.layer1 = nn.Dense(self.d_model)
        self.layer2 = nn.Dense(self.d_output)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        x = nn.gelu(x)
        return self.layer2(x)


class RetrievalModel(nn.Module):
    """S5 Retrieval classification model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, constructing 4 features which are fed into a MLP,
    and a softmax operation. Note that unlike the standard classification model above,
    the apply function of this model operates directly on the batch of data (instead of calling
    vmap on this model).
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
    """

    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and the retrieval decoder. Note that here we
        vmap over the stacked encoder model to work well with the retrieval decoder that
        operates directly on the batch.
        """
        BatchEncoderModel = nn.vmap(
            StackedEncoderModel,
            in_axes=(0, 0),
            out_axes=0,
            variable_axes={
                "params": None,
                "dropout": None,
                "batch_stats": None,
                "cache": 0,
                "prime": None,
            },
            split_rngs={"params": False, "dropout": True},
            axis_name="batch",
        )

        self.encoder = BatchEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        BatchRetrievalDecoder = nn.vmap(
            RetrievalDecoder,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.decoder = BatchRetrievalDecoder(
            d_model=self.d_model, d_output=self.d_output
        )

    def __call__(
        self, input, integration_timesteps
    ):  # input is a tuple of x and lengths
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence. The encoded features are constructed as in
        Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
        Args:
             input (float32, int32): tuple of input sequence and prepadded sequence lengths
                input sequence is of shape (2*bsz, L, d_input) (includes both documents) and
                lengths is (2*bsz,)
        Returns:
            output (float32): (d_output)
        """
        x, lengths = input  # x is 2*bsz*seq_len*in_dim, lengths is: (2*bsz,)
        x = self.encoder(
            x, integration_timesteps
        )  # The output is: 2*bszxseq_lenxd_model
        outs = batch_masked_meanpool(x, lengths)  # Avg non-padded values: 2*bszxd_model
        outs0, outs1 = np.split(outs, 2)  # each encoded_i is bszxd_model
        features = np.concatenate(
            [outs0, outs1, outs0 - outs1, outs0 * outs1], axis=-1
        )  # bszx4*d_model
        out = self.decoder(features)
        return nn.log_softmax(out, axis=-1)


class DVSFrameModel(nn.Module):
    """S5 Retrieval classification model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, constructing 4 features which are fed into a MLP,
    and a softmax operation. Note that unlike the standard classification model above,
    the apply function of this model operates directly on the batch of data (instead of calling
    vmap on this model).
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
    """

    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    use_cnn: bool = False

    def setup(self):
        """
        Initializes the S5 stacked encoder and the retrieval decoder. Note that here we
        vmap over the stacked encoder model to work well with the retrieval decoder that
        operates directly on the batch.
        """
        BatchEncoderModel = nn.vmap(
            StackedEncoderModel,
            in_axes=(0, 0),
            out_axes=0,
            variable_axes={
                "params": None,
                "dropout": None,
                "batch_stats": None,
                "cache": 0,
                "prime": None,
            },
            split_rngs={"params": False, "dropout": True},
            axis_name="batch",
        )

        self.encoder = BatchEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        BatchRetrievalDecoder = nn.vmap(
            RetrievalDecoder,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )

        self.decoder = BatchRetrievalDecoder(
            d_model=self.d_model, d_output=self.d_output
        )

    def __call__(
        self, input, integration_timesteps
    ):  # input is a tuple of x and lengths
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence. The encoded features are constructed as in
        Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
        Args:
             input (float32, int32): tuple of input sequence and prepadded sequence lengths
                input sequence is of shape (2*bsz, L, d_input) (includes both documents) and
                lengths is (2*bsz,)
        Returns:
            output (float32): (d_output)
        """
        x, lengths = input  # x is 2*bsz*seq_len*in_dim, lengths is: (2*bsz,)
        x = self.encoder(
            x, integration_timesteps
        )  # The output is: 2*bszxseq_lenxd_model
        outs = batch_masked_meanpool(x, lengths)  # Avg non-padded values: 2*bszxd_model
        outs0, outs1 = np.split(outs, 2)  # each encoded_i is bszxd_model
        features = np.concatenate(
            [outs0, outs1, outs0 - outs1, outs0 * outs1], axis=-1
        )  # bszx4*d_model
        out = self.decoder(features)
        return nn.log_softmax(out, axis=-1)
