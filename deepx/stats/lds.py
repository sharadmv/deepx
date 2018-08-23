import numpy as np
from .. import T

from .common import ExponentialFamily
from .gaussian import Gaussian

class LDS(ExponentialFamily):

    def __init__(self, parameters):
        super(LDS, self).__init__(self.construct_natparam(parameters), 'natural')

    def get_param_dim(self):
        return 3

    def expected_value(self):
        raise NotImplementedError

    def log_h(self, x):
        return T.zeros(T.shape(x)[:-1])

    def sufficient_statistics(self, x):
        raise NotImplementedError

    def expected_sufficient_statistics(self):
        raise NotImplementedError

    def sample(self, num_samples=1):
        raise NotImplementedError

    @classmethod
    def regular_to_natural(cls, l):
        raise NotImplementedError

    @classmethod
    def natural_to_regular(cls, eta):
        raise NotImplementedError

    def log_likelihood(self, x):
        raise NotImplementedError

    def log_z(self):
        natparam = self.get_parameters('natural')
        leading_dim = T.shape(natparam)[:-2]
        return T.zeros(leading_dim)

    def construct_natparam(self, parameters):
        if len(parameters) == 3:
            (A, B, Q), prior, actions = parameters
            potentials = None
        elif len(parameters) == 4:
            (A, B, Q), prior, potentials, actions = parameters
        Q_inv = T.matrix_inverse(Q)
        Q_inv_A = T.matrix_solve(Q, A)
        Q_inv_B = T.matrix_solve(Q, B)

        B_shape = T.shape(B)
        H, ds, da = B_shape[0], B_shape[1], B_shape[2]
        H = H + 1

        A, B, Q, Q_inv, Q_inv_A, Q_inv_B = tuple(map(
            lambda x: T.core.pad(x, [[0, 1], [0, 0], [0, 0]])
        , (A, B, Q, Q_inv, Q_inv_A, Q_inv_B)))

        prior_eta1, prior_eta2 = Gaussian.unpack(prior.get_parameters('natural'))
        prior_natparam = T.core.pad(
          vs([
              hs([prior_eta1, T.zeros([ds, ds]), prior_eta2[..., None]]),
              T.zeros([ds, 2 * ds + 1]),
              hs([prior_eta2[None], T.zeros([1, ds]), T.to_float(-prior.log_z() - prior.log_h(prior_eta2))[None, None]]),
          ])[None]
        , [[0, H - 1], [0, 0], [0, 0]])
        QBa = T.einsum('tab,tb->ta', Q_inv_B, actions)[..., None]
        AQBa = T.einsum('tba,tbc->tac', A, QBa)
        dynamics_natparam = 0.5 * (
            vs([
                hs([-T.einsum('tba,tbc->tac', A, Q_inv_A), t(Q_inv_A), -AQBa]),
                hs([Q_inv_A                              , -Q_inv    , QBa]),
                hs([-t(AQBa)                             , t(QBa)    , -T.einsum('ta,tba,tbc->tc', actions, B, AQBa)[..., None]
                                                                       -(T.logdet(Q) + T.to_float(ds) * T.log(2 * np.pi))[..., None, None]]),
            ])
        )
        if potentials is None:
            return prior_natparam + dynamics_natparam
        else:
            potential_eta1, potential_eta2 = Gaussian.unpack(potentials.get_parameters('natural'))
            potentials_natparam = (
                vs([
                  hs([potential_eta1, T.zeros([H, ds, ds]), potential_eta2[..., None]]),
                  T.zeros([H, ds, 2 * ds + 1]),
                  hs([potential_eta2[..., None, :], T.zeros([H, 1, ds]), T.to_float(-potentials.log_z() - potentials.log_h(potential_eta2))[..., None, None]]),
                ])
            )
            return prior_natparam + dynamics_natparam + potentials_natparam

hs = lambda x: T.concat(x, -1)
vs = lambda x: T.concat(x, -2)
t  = lambda x: T.matrix_transpose(x)
