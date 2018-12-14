import numpy as np
from .. import T
import tensorflow as tf

from .common import ExponentialFamily
from .gaussian import Gaussian

class LDS(ExponentialFamily):

    def __init__(self, parameters, parameter_type='internal'):
        if parameter_type == 'internal':
            natparam = self._construct_natparam(parameters)
            self.needs_internal_params = False
        elif parameter_type == 'natural':
            natparam = parameters
            self.needs_internal_params = True
        else:
            raise NotImplementedError
        super(LDS, self).__init__(natparam, 'natural')
        if parameter_type == 'internal':
            self._parameter_cache['internal'] = parameters
        self._cached = {}

    def set_internal_params(self, params):
        self._parameter_cache['internal'] = params

    def get_param_dim(self):
        return 3

    def log_h(self, x):
        return T.zeros(T.shape(x)[:-1])

    def sufficient_statistics(self, x):
        raise NotImplementedError

    def log_z(self):
        if not 'log_z' in self._cached:
            natparam = self.get_parameters('natural')
            N, H, ds = T.shape(natparam)[0], natparam.get_shape()[1], (T.shape(natparam)[2] - 1) // 2
            pred_potential = T.zeros((N, ds+1, ds+1))

            for t_ in range(H):
                pred_potential_expand = (
                    vs([
                      hs([pred_potential[:, :ds, :ds], T.zeros((N, ds, ds)), pred_potential[:, :ds, ds:]]),
                      T.zeros((N, ds, 2 * ds + 1)),
                      hs([pred_potential[:, ds:, :ds], T.zeros((N, 1, ds)), pred_potential[:, ds:, ds:]]),
                    ])
                )
                filter_natparam = natparam[:, t_] + pred_potential_expand
                A, B, C = filter_natparam[:, :ds, :ds], filter_natparam[:, :ds, ds:], filter_natparam[:, ds:, ds:]
                schur_comp = C - T.matmul(t(B), T.matrix_solve(A, B))
                norm = 0.5 * T.logdet(-2 * filter_natparam[:, :ds, :ds]) + T.to_float(ds / 2) * T.log(2 * np.pi)
                pred_potential = schur_comp - T.matrix_diag(T.concat([T.zeros(ds), T.ones(1)])) * norm[..., None, None]

            self._cached['log_z'] = T.sum(pred_potential[:, -1, -1])
        return self._cached['log_z']

    def expected_sufficient_statistics(self):
        if not 'expected_sufficient_statistics' in self._cached:
            ess = T.gradients(self.log_z(), self.get_parameters('natural'))[0]
            self._cached['expected_sufficient_statistics'] = (ess + T.matrix_transpose(ess)) / 2.0
        return self._cached['expected_sufficient_statistics']

    def expected_value(self):
        ess = self.expected_sufficient_statistics()
        ds = (T.shape(ess)[-1] - 1) // 2
        return ess[..., :ds, -1]

    def smooth(self):
        ess = self.expected_sufficient_statistics()
        ds = (T.shape(ess)[-1] - 1) // 2
        mu = ess[..., :ds, -1]
        sig = ess[..., :ds, :ds] - T.outer(mu, mu)
        return Gaussian([sig, mu])

    def filter(self):
        parameters = self.get_parameters('internal')
        assert len(parameters) == 4, 'missing state node potentials?'
        (A, B, Q), prior, potentials, actions = parameters
        prior_natparams = Gaussian.unpack(prior.get_parameters('natural'))
        A_inv, Q_inv = T.matrix_inverse(A), T.matrix_inverse(Q)
        A, B, Q, Q_inv = tuple(map(
            lambda x: T.core.pad(x, [[0, 1], [0, 0], [0, 0]]),
            (A, B, Q, Q_inv)
        ))
        N, H, ds = T.shape(potentials[1])[0], T.shape(potentials[1])[1], T.shape(potentials[1])[2]
        A_inv = T.concat([A_inv, T.eye(ds)[None]], axis=0)

        J_22 = Q_inv
        J_12 = -T.einsum('tba,tbc->tac', A, J_22)
        J_21 = T.matrix_transpose(J_12)
        J_11 = -T.einsum('tab,tbc->tac', J_12, A)
        h2 = T.einsum('ita,tba,tbc->tic', actions, B, J_22)
        h1 = -T.einsum('tia,tab->tib', h2, A)
        self._cached['info_params'] = (J_11, J_12, J_22, h1, h2)

        potentials = (-2 * T.transpose(potentials[0], [1, 0, 2, 3]), T.transpose(potentials[1], [1, 0, 2]))
        actions = T.transpose(actions, [1, 0, 2])
        prior_natparams[0] = T.tile(prior_natparams[0][None], [N, 1, 1])
        prior_natparams[1] = T.tile(prior_natparams[1][None], [N, 1])

        def kalman_filter(previous, potential):
            t_, _, prev = previous
            J_tt = prev[0] + potential[0]
            h_tt = prev[1] + potential[1]
            M = T.einsum('ba,ibc,cd->iad', A_inv[t_], J_tt, A_inv[t_])
            inv_term = T.eye(ds) + T.einsum('iab,bc->iac', M, Q[t_])
            J_t1_t = T.matrix_solve(inv_term, M)
            h_t1_t = T.einsum('iab,ib->ia',
                              T.matrix_solve(inv_term, T.tile(T.transpose(A_inv[t_])[None], [N, 1, 1])),
                              h_tt) + \
                    T.einsum('iab,bc,ic->ia', J_t1_t, B[t_], actions[t_])

            return t_ + 1, (J_tt, h_tt), (J_t1_t, h_t1_t)

        _, filtered, _ = T.scan(kalman_filter, potentials,
                                (0,
                                 (T.zeros([N, ds, ds]), T.zeros([N, ds])),
                                 (-2 * prior_natparams[0], prior_natparams[1]))
        )
        filtered = (T.transpose(filtered[0], [1, 0, 2, 3]), T.transpose(filtered[1], [1, 0, 2]))
        self._cached['filtered'] = filtered

        return Gaussian([
            T.matrix_inverse(filtered[0]),
            T.matrix_solve(filtered[0], filtered[1][..., None])[..., 0]
        ])

    def sample(self, num_samples=1):
        filter_dist = self.filter()
        filter_sample = filter_dist.sample(num_samples=num_samples)
        sample = filter_sample[..., -1:, :]
        J11, J12, J22, h1, h2 = self._cached['info_params']
        Jtt, htt = self._cached['filtered']
        H = self._parameter_cache['internal'][0][0].get_shape()[0]
        for t_ in range(H)[::-1]:
            J_t = T.tile((Jtt[:, t_] + J11[t_])[None], [num_samples, 1, 1, 1])
            h_t = htt[:, t_] + h1[t_] - T.einsum('nia,ab->nib', sample[..., 0, :], t(J12)[t_])
            dist_t = Gaussian([
                T.matrix_inverse(J_t), T.matrix_solve(J_t, h_t[..., None])[..., 0]
            ])
            sample = T.concat([dist_t.sample()[0][..., None, :], sample], axis=-2)
        return sample

    @classmethod
    def regular_to_natural(cls, l):
        raise NotImplementedError

    @classmethod
    def natural_to_regular(cls, eta):
        raise NotImplementedError

    def log_likelihood(self, x):
        raise NotImplementedError

    def _construct_natparam(self, parameters):
        if len(parameters) == 3:
            (A, B, Q), prior, actions = parameters
            potentials = None
        elif len(parameters) == 4:
            (A, B, Q), prior, potentials, actions = parameters
        Q_inv = T.matrix_inverse(Q)
        Q_inv_A = T.matrix_solve(Q, A)
        Q_inv_B = T.matrix_solve(Q, B)
        logdetQ = T.logdet(Q)

        B_shape = T.shape(B)
        H, ds, da = B_shape[0], B_shape[1], B_shape[2]
        H = H + 1
        N = T.shape(actions)[0]

        A, B, Q, Q_inv, Q_inv_A, Q_inv_B = tuple(map(
            lambda x: T.core.pad(x, [[0, 1], [0, 0], [0, 0]]),
            (A, B, Q, Q_inv, Q_inv_A, Q_inv_B)
        ))
        logdetQ = T.concat([logdetQ, T.zeros(1)])

        prior_eta1, prior_eta2 = Gaussian.unpack(prior.get_parameters('natural'))
        prior_natparam = T.core.pad(
            vs([
                hs([prior_eta1, T.zeros([ds, ds]), 0.5 * prior_eta2[..., None]]),
                T.zeros([ds, 2 * ds + 1]),
                0.5 * hs([prior_eta2[None], T.zeros([1, ds]), T.to_float(-prior.log_z())[None, None]]),
            ])[None],
            [[0, H - 1], [0, 0], [0, 0]]
        )
        h2 = T.einsum('tab,itb->ita', Q_inv_B, actions)[..., None]
        h1 = -T.einsum('tba,itbc->itac', A, h2)
        J11 = -T.tile(T.einsum('tba,tbc->tac', A, Q_inv_A)[None], [N, 1, 1, 1])
        J12 = T.tile(t(Q_inv_A)[None], [N, 1, 1, 1])
        J22 = -T.tile(Q_inv[None], [N, 1, 1, 1])
        dynamics_natparam = 0.5 * (
            vs([
                hs([J11,    J12,   h1]),
                hs([t(J12), J22,   h2]),
                hs([t(h1),  t(h2), -T.einsum('ita,tba,itbc->itc', actions, B, h2)[..., None]
                                   - logdetQ[..., None, None]]),
            ])
        )
        if potentials is None:
            return (prior_natparam + dynamics_natparam)
        else:
            N = T.shape(potentials[0])[0]
            potentials_natparam = (
                vs([
                  hs([potentials[0], T.zeros([N, H, ds, ds]), 0.5 * potentials[1][..., None]]),
                  T.zeros([N, H, ds, 2 * ds + 1]),
                  0.5 * hs([potentials[1][..., None, :], T.zeros([N, H, 1, ds]), T.to_float(-potentials[2])[..., None, None]]),
                ])
            )
            return prior_natparam + dynamics_natparam + potentials_natparam


hs = lambda x: T.concat(x, -1)
vs = lambda x: T.concat(x, -2)
t  = lambda x: T.matrix_transpose(x)
