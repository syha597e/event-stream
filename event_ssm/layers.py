from flax import linen as nn
import jax
from functools import partial


class EventPooling(nn.Module):
    stride: int = 1
    mode: str = "last"
    eps: float = 1e-6

    def __call__(self, x, integration_timesteps):
        """
        Compute the pooled (L/stride)xH output given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
             integration_timesteps (float32): the integration timesteps for the SSM
        Returns:
            output sequence (float32): (L/stride, d_model)
        """
        if self.stride == 1:
            raise ValueError("Stride 1 not supported for pooling")
        else:
            remaining_timesteps = (len(integration_timesteps) // self.stride) * self.stride
            new_integration_timesteps = integration_timesteps[:remaining_timesteps].reshape(-1, self.stride).sum(axis=1)
            x = x[:remaining_timesteps]
            d_model = x.shape[-1]
            if self.mode == 'last':
                x = x[::self.stride]
                return x, new_integration_timesteps
            elif self.mode == 'avgpool':
                x = x.reshape(-1, self.stride, d_model).mean(axis=1)
                return x, new_integration_timesteps
            elif self.mode == 'timepool':
                weight = integration_timesteps[:remaining_timesteps, None] + self.eps
                x = (x * weight).reshape(-1, self.stride, d_model).sum(axis=1)
                x = x / weight.reshape(-1, self.stride, 1).sum(axis=1)
                return x, new_integration_timesteps
            else:
                raise NotImplementedError("Pooling mode: {} not implemented".format(self.stride))


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    discretization: str
    dropout: float
    d_model_in: int
    d_model_out: int
    d_ssm: int
    block_size: int
    activation: str = "gelu"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    pooling_stride: int = 1
    pooling_mode: str = "last"

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(H_in=self.d_model_in, H_out=self.d_model_out, P=self.d_ssm, block_size=self.block_size, step_rescale=self.step_rescale, discretization=self.discretization, stride=self.pooling_stride, pooling_mode=self.pooling_mode)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model_out)
            self.out2 = nn.Dense(self.d_model_out)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model_out)

        if self.d_model_in != self.d_model_out:
            self.skip_connection = nn.Dense(self.d_model_out)

        if self.batchnorm:
            self.norm = nn.BatchNorm(momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0]
        )

        self.pool = EventPooling(stride=self.pooling_stride, mode=self.pooling_mode)

    def __call__(self, x, integration_timesteps, train: bool):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x

        if self.prenorm:
            if self.batchnorm:
                x = self.norm(x, use_running_average=not train)
            else:
                x = self.norm(x)

        x = self.seq(x, integration_timesteps)

        apply_dropout = partial(self.drop, deterministic=not train)
        if self.activation in ["full_glu"]:
            x = apply_dropout(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = apply_dropout(x)
        elif self.activation in ["half_glu1"]:
            x = apply_dropout(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = apply_dropout(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = apply_dropout(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = apply_dropout(x)
        elif self.activation in ["gelu"]:
            x = apply_dropout(nn.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        if self.pooling_stride > 1:
            skip, integration_timesteps = self.pool(skip, integration_timesteps)

        if self.d_model_in != self.d_model_out:
            skip = self.skip_connection(skip)

        # apply skip connection
        x = skip + x

        if not self.prenorm:
            if self.batchnorm:
                x = self.norm(x, use_running_average=not train)
            else:
                x = self.norm(x)

        return x, integration_timesteps
