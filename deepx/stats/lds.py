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
        return T.gradients(self.log_z(), self.get_parameters('natural'))

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
        #TODO: need to support batches?
        def step(prev, elems):
            t_, pred_potential = prev
            pred_potential_expand = (
                vs([
                  hs([pred_potential[:ds, :ds], T.zeros((ds, ds)), pred_potential[:ds, ds:]]),
                  T.zeros((ds, 2 * ds + 1)),
                  hs([pred_potential[ds:, :ds], T.zeros((1, ds)), pred_potential[ds:, ds:]]),
                ])
            )
            filter_natparam = natparam[t_, :, :] + pred_potential_expand
            A, B, C = filter_natparam[:ds, :ds], filter_natparam[:ds, ds:], filter_natparam[ds:, ds:]
            schur_comp = C - T.matmul(t(B), T.matrix_solve(A, B))
            norm = 0.5 * T.logdet(-2 * filter_natparam[:ds, :ds]) + T.to_float(ds / 2) * T.log(2 * np.pi)
            next_pred_potential = schur_comp - T.matrix_diag(T.concat([T.zeros(ds), T.ones(1)])) * norm
            next_pred_potential.set_shape(pred_potential.get_shape())
            return t_ + 1, next_pred_potential

        natparam = self.get_parameters('natural')
        H, ds = T.shape(natparam)[0], (T.shape(natparam)[1] - 1) // 2
        _, pred_potentials = T.scan(step, (T.zeros(H, dtype=T.int32), T.zeros((H, ds+1, ds+1))),
                                 (0, T.zeros((ds+1, ds+1))))
        return pred_potentials[-1, -1, -1]

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
