import numpy as np

import reservoirpy as rpy
from reservoirpy.node import Node
from reservoirpy.mat_gen import bernoulli


class Larger(Node):
    """Model of an optoelectronic reservoir, proposed by Larger et al. [1]_

    Parameters
    ----------
    N : int
        Number of virtual units
    tau : float
        Delay induced by the optical fiber
    kappa : float
        Input gain (input scaling in ESNs)
    rho : float
        Recurrent gain (~ spectral radius in ESNs)
    phi : float
        Phase bias
    beta : float
        Non-linear gain
    epsilon : float
        Low-pass filter parameter

    References
    ----------
    .. [1] Larger, L., Soriano, M. C., Brunner, D., Appeltant, L., Gutiérrez,
        J. M., Pesquera, L., ... & Fischer, I. (2012). Photonic information
        processing beyond Turing: an optoelectronic implementation of reservoir computing.
        Optics express, 20(3), 3241-3249.
    """

    def __init__(
        self, N, tau, kappa, rho, phi, beta, epsilon, input_mask=bernoulli, input_dim=None, **kwargs
    ):

        super(Larger, self).__init__(
            hypers=dict(
                N=N,
                tau=tau,
                kappa=kappa,
                rho=rho,
                phi=phi,
                beta=beta,
                epsilon=epsilon,
                dt=tau / N,
            ),
            params={"input_mask": input_mask},
            forward=Larger._forward,
            initializer=Larger._initialize,
            input_dim=input_dim,
            output_dim=N,
            **kwargs,
        )

    def _forward(self, x):
        # Parameters
        dt = self.dt
        N = self.N
        # States
        past_state = self._state.T
        new_state = np.zeros(self._state.shape).T

        if self.input_mask is not False:
            x = x @ self.input_mask

        for k in range(1, N):
            k1 = self.f(
                u=x[:, k-1],
                past=past_state[k-1],
                x=new_state[k-1],
            )
            k2 = self.f(
                u=(x[:, k - 1] + x[:, k]) / 2,
                past=(past_state[k-1] + past_state[k]) / 2,
                x=new_state[k - 1] - dt * k1 / 2,
            )
            k3 = self.f(
                u=(x[:, k - 1] + x[:, k]) / 2,
                past=(past_state[k-1] + past_state[k]) / 2,
                x=new_state[k-1] - dt * k2 / 2,
            )
            k4 = self.f(
                u=x[:, k],
                past=past_state[k],
                x=new_state[k-1] - dt * k3,
            )
            new_state[k] = new_state[k-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

        return new_state.T

    def _initialize(self, x, y=None):
        # Only 1 input dimension here
        assert len(x.shape) == 2
        if isinstance(x, list):
            self.set_input_dim(x[0].shape[-1])
        else:
            self.set_input_dim(x.shape[-1])
        

        if self.input_mask is None:
            assert self.input_dim == self.N
        else:
            if callable(self.input_mask):
                self.input_mask = self.input_mask(
                    self.input_dim, self.N
                )
            else:
                assert self.input_mask.shape == (self.input_dim, self.N)

        self._state = np.zeros(self.N)

        if self.input_mask is not False:
            x = x @ self.input_mask

        self._state[0] = x[0, 0]
        for k in range(1, self.N):
            incr = self.f(u=x[0, k - 1], past=0.0, x=self._state[k - 1])
            self._state[k] = self._state[k - 1] + incr

        return

    def f(self, u, past, x):
        nonlinear = np.square(np.sin(self.kappa * u + self.rho * past + self.phi))
        return (self.beta * nonlinear - x) * self.dt / self.epsilon
