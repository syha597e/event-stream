from flax import linen as nn
import jax


class EventPooling(nn.Module):
    stride: int = 1
    mode: str = "last"

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
        if self.mode == 'last':
            x = x[::self.stride]
            remaining_timesteps = (len(integration_timesteps) // self.stride) * self.stride
            integration_timesteps = integration_timesteps[:remaining_timesteps].reshape(-1, self.stride).sum(axis=1)
            return x, integration_timesteps
        elif self.mode == 'mean':
            x = x.reshape(-1, self.stride, x.shape[-1]).mean(axis=1)
            remaining_timesteps = (len(integration_timesteps) // self.stride) * self.stride
            integration_timesteps = integration_timesteps[:remaining_timesteps].reshape(-1, self.stride).sum(axis=1)
            return x, integration_timesteps
        else:
            raise NotImplementedError("Stride: {} not implemented".format(self.stride))


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
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
    dropout: float
    d_model: int
    d_ssm: int
    block_size: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    stride: int = 1
    pooling_mode: str = "last"

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(H=self.d_model, P=self.d_ssm, block_size=self.block_size, step_rescale=self.step_rescale, discretization=self.discretization, stride=self.stride, pooling_mode=self.pooling_mode)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model * self.seq.state_expansion)
            self.out2 = nn.Dense(self.d_model * self.seq.state_expansion)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model * self.seq.state_expansion)

        if self.stride > 1:
            self.skip_connection = nn.Dense(self.d_model * self.seq.state_expansion)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

        self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x

        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x, integration_timesteps)

        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(self.out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        if self.stride > 1:
            skip, integration_timesteps = self.pool(skip, integration_timesteps)
            skip = self.skip_connection(skip)

        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x, integration_timesteps
