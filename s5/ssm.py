from functools import partial
import jax
import jax.numpy as np
from jax.scipy.linalg import block_diag

from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal, make_DPLR_HiPPO

from .layers import EventPooling


# Discretization functions
def discretize_bilinear(Lambda, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    gamma_bar = (BL * Delta)
    return Lambda_bar, gamma_bar


def discretize_zoh(Lambda, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = (1/Lambda * (Lambda_bar-Identity))
    return Lambda_bar, gamma_bar


def discretize_dirac(Lambda, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        with dirac delta input spikes.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Lambda_bar = np.exp(Lambda * Delta)
    gamma_bar = 1.0
    return Lambda_bar, gamma_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_elements, Bu_elements, C_tilde, conj_sym, stride=1):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    xs = xs[::stride]

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class S5SSM(nn.Module):
    H: int
    P: int
    block_size: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    step_rescale: float = 1.0
    stride: int = 1
    pooling_mode: str = "last"

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """
        # if stride is larger than 1, set state_expansion to 2
        self.state_expansion = 1 if self.stride == 1 else 2

        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(self.block_size)

        blocks = self.P // self.block_size
        block_size = self.block_size // 2 if self.conj_sym else self.block_size
        local_P = self.P // 2 if self.conj_sym else self.P

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

        print(f"SSM: {self.H} -> {self.P} -> {self.H * self.state_expansion} (using pooling mode {self.pooling_mode})")

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: Lambda.real, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: Lambda.imag, (None,))

        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (self.P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init, rng, shape, Vinv),
                            B_shape)

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H * self.state_expansion, self.P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H * self.state_expansion, self.P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            C = self.param("C", C_init, (self.H * self.state_expansion, local_P, 2))
            self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            self.C = self.param("C",
                                lambda rng, shape: init_CV(C_init, rng, shape, V),
                                C_shape)

            self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H * self.state_expansion,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (local_P, self.dt_min, self.dt_max))

        # pooling layer
        self.pool = EventPooling(stride=self.stride, mode=self.pooling_mode)

        # Discretize
        if self.discretization in ["zoh"]:
            self.discretize_fn = discretize_zoh
        elif self.discretization in ["bilinear"]:
            self.discretize_fn = discretize_bilinear
        elif self.discretization in ["dirac"]:
            self.discretize_fn = discretize_dirac
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, input_sequence, integration_timesteps):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """

        # discretize on the fly
        B = self.B[..., 0] + 1j * self.B[..., 1]

        def discretize_and_project_inputs(u, _timestep):
            step = self.step_rescale * np.exp(self.log_step[:, 0])
            Lambda_bar, gamma_bar = self.discretize_fn(self.Lambda, step * _timestep)
            Bu = gamma_bar * (B @ u)
            return Lambda_bar, Bu

        Lambda_bar_elements, Bu_bar_elements = jax.vmap(discretize_and_project_inputs)(input_sequence, integration_timesteps)

        ys = apply_ssm(
            Lambda_bar_elements,
            Bu_bar_elements,
            self.C_tilde,
            self.conj_sym,
            stride=self.stride
        )

        if self.stride > 1:
            input_sequence, _ = self.pool(input_sequence, integration_timesteps)
            input_sequence = np.repeat(input_sequence, self.state_expansion, axis=1)

        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


def init_S5SSM(
        C_init,
        dt_min,
        dt_max,
        conj_sym,
        clip_eigs,
):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   C_init=C_init,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs
                   )
