from functools import partial

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
        self, N, tau, kappa, rho, phi, beta, epsilon, input_mask=bernoulli, input_dim=None, seed=None, **kwargs
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
            initializer=partial(Larger._initialize, seed=seed),
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

        k1 = self.f(
            u=self._last_input,
            past=self._last_state,
            x=past_state[-1],
        )
        k2 = self.f(
            u=(self._last_input + x[:, 0]) / 2,
            past=(self._last_state + past_state[-1]) / 2,
            x=past_state[-1] - dt * k1 / 2,
        )
        k3 = self.f(
            u=(self._last_input + x[:, 0]) / 2,
            past=(self._last_state + past_state[-1]) / 2,
            x=past_state[-1] - dt * k2 / 2,
        )
        k4 = self.f(
            u=x[:, 0],
            past=past_state[0],
            x=past_state[-1] - dt * k3,
        )
        new_state[0] = past_state[-1] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        # new_state[0] = 0.0

        for k in range(1, N):
            k1 = self.f(
                u=x[:, k-1],
                past=past_state[k-1],
                x=new_state[k-1],
            )
            k2 = self.f(
                u=(x[:, k - 1] + x[:, k]) / 2,
                past=(past_state[k-1] + past_state[k]) / 2,
                x=new_state[k-1] - dt * k1 / 2,
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
            new_state[k] = new_state[k-1] + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

        # trick to get the last reservoir input for the first
        # neuron of the following timestep
        self._last_input = x[:, -1]
        self._last_state = past_state[-1]

        return new_state.T

    def _initialize(self, x, y=None, seed=None):
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
                    self.input_dim, self.N,
                    seed=seed,
                )
            else:
                assert self.input_mask.shape == (self.input_dim, self.N)

        self._state = np.zeros(self.N)
        self._last_input = np.array([0.0])
        self._last_state = 0.0

    def f(self, u, past, x):
        """
            beta*sin^2[ kappa*u_in + rho x(t-tau) + phi) - x(t) ] / epsilon
            (tel que dx/dt = f(t, y) )
        """
        nonlinear = np.square(np.sin(self.kappa * u + self.rho * past + self.phi))
        return (self.beta * nonlinear - x) / self.epsilon
