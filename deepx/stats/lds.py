import numpy as np
from .. import T
import tensorflow as tf

from .common import ExponentialFamily
from .gaussian import Gaussian

class LDS(ExponentialFamily):

    def __init__(self, parameters, parameter_type='internal'):
        super(LDS, self).__init__(parameters, parameter_type=parameter_type)
        self.cache = {}

    def get_param_dim(self):
        return 3

    def log_h(self, x):
        return T.zeros(T.shape(x)[:-1])

    def sufficient_statistics(self, x):
        raise NotImplementedError

    def log_z(self):
        if not 'log_z' in self.cache:
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

            self.cache['log_z'] = T.sum(pred_potential[:, -1, -1])
        return self.cache['log_z']

    def expected_sufficient_statistics(self):
        if not 'ess' in self.cache:
            ess = T.gradients(self.log_z(), self.get_parameters('natural'))[0]
            self.cache['ess'] = (ess + T.matrix_transpose(ess)) / 2.0
        return self.cache['ess']

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
        (nh_Q_inv, Q_inv_AB, nh_AB_Q_inv_AB, nh_logdet_Q), prior, q_S, actions, horizon = parameters

        actions = T.transpose(actions, [1, 0, 2])

        if q_S is not None:
            natparam = Gaussian.unpack(q_S.get_parameters('natural'))
            J, h, log_z = natparam[0], natparam[1], q_S.log_z()
            J = -2 * T.transpose(J, [1, 0, 2, 3])
            h = T.transpose(h, [1, 0, 2])
            log_z = T.transpose(log_z, [1, 0])
            potentials = [J, h, log_z]
        else:
            raise Exception('Missing state node potentials')

        prior_natparams = Gaussian.unpack(prior.get_parameters('natural'))
        ds = T.shape(nh_Q_inv)[-1]

        nh_Q_inv = T.concatenate([
            nh_Q_inv,
            -0.5 * T.eye(ds)[None]
        ], 0)
        Q_inv_AB, nh_AB_Q_inv_AB = tuple(map(
            lambda x: T.core.pad(x, [[0, 1], [0, 0], [0, 0]]),
            (Q_inv_AB, nh_AB_Q_inv_AB)
        ))

        N = T.shape(potentials[0])[1]
        Q_inv = -2 * nh_Q_inv

        Q_inv_A = Q_inv_AB[..., :ds]
        Q_inv_B = Q_inv_AB[..., ds:]
        A_Q_inv_A = -2 * nh_AB_Q_inv_AB[..., :ds, :ds]
        A_Q_inv_B = -2 * nh_AB_Q_inv_AB[..., :ds, ds:]
        B_Q_inv_B = -2 * nh_AB_Q_inv_AB[..., ds:, ds:]

        J_22 = Q_inv
        J_21 = -Q_inv_A
        AB = T.einsum('tab,tbc->tac', T.matrix_inverse(Q_inv), Q_inv_AB)
        A, B = AB[..., :ds], AB[..., ds:]
        J_12 = T.matrix_transpose(J_21)
        J_11 = A_Q_inv_A

        prior_natparams = (
            T.tile(prior_natparams[0][None], [N, 1, 1]),
            T.tile(prior_natparams[1][None], [N, 1])
        )
        def kalman_filter(previous, potential):
            t, _, prev = previous
            J_tt = prev[0] + potential[0]
            h_tt = prev[1] + potential[1]

            J_ = J_22[t][None] - T.einsum('ab,nbc,cd->nad', J_21[t], T.matrix_inverse(J_tt + J_11[t][None]), J_12[t])
            h_ = T.einsum('nab,nb->na', J_,
                          T.einsum('ab,nb->na', A[t], T.matrix_solve(J_tt, h_tt[..., None])[..., 0])
                          + T.einsum('ab,nb->na', B[t], actions[t])
                          )

            return t + 1, (J_tt, h_tt), (J_, h_)

        _, filtered, _ = T.scan(kalman_filter, potentials,
                                (0,
                                 (T.zeros([N, ds, ds]), T.zeros([N, ds])),
                                 (-2 * prior_natparams[0], prior_natparams[1]))
        )
        filtered = (T.transpose(filtered[0], [1, 0, 2, 3]), T.transpose(filtered[1], [1, 0, 2]))

        return Gaussian(Gaussian.pack([
            -0.5 * filtered[0],
            filtered[1],
        ]), 'natural')

    def sample(self, num_samples=1):
        if ('sample', num_samples) not in self.cache:
            filter_dist = self.filter()
            (nh_Q_inv, Q_inv_AB, nh_AB_Q_inv_AB, nh_logdet_Q), prior, q_S, actions, horizon = self.get_parameters('internal')

            actions = actions[:, :-1]

            Q_inv = -2 * nh_Q_inv
            ds = T.shape(Q_inv)[-1]

            Q_inv_A = Q_inv_AB[..., :ds]
            Q_inv_B = Q_inv_AB[..., ds:]
            logdetQ = -2 * nh_logdet_Q
            A_Q_inv_A = -2 * nh_AB_Q_inv_AB[..., :ds, :ds]
            A_Q_inv_B = -2 * nh_AB_Q_inv_AB[..., :ds, ds:]
            B_Q_inv_B = -2 * nh_AB_Q_inv_AB[..., ds:, ds:]
            AB = T.einsum('tab,tbc->tac', T.matrix_inverse(Q_inv), Q_inv_AB)
            A, B = AB[..., :ds], AB[..., ds:]

            J_22 = Q_inv
            J_21 = -Q_inv_A
            J_12 = T.matrix_transpose(J_21)
            J_11 = A_Q_inv_A
            h2 = T.einsum('tab,itb->ita', Q_inv_B, actions)[..., None]
            h1 = -T.einsum('tab,itb->ita', A_Q_inv_B, actions)

            filter_sample = filter_dist.sample(num_samples=num_samples)

            samples = [filter_sample[..., -1:, :]]
            n2_Jtt, htt = Gaussian.unpack(filter_dist.get_parameters('natural'))
            Jtt = -2 * n2_Jtt
            H = horizon
            for t_ in range(H - 1)[::-1]:
                J_t = T.tile((Jtt[:, t_] + J_11[t_])[None], [num_samples, 1, 1, 1])
                h_t = htt[:, t_] + h1[:, t_] - T.einsum('nia,ab->nib', samples[0][..., 0, :], t(J_12)[t_])
                dist_t = Gaussian([
                    T.matrix_inverse(J_t), T.matrix_solve(J_t, h_t[..., None])[..., 0]
                ])
                samples.insert(0, dist_t.sample()[0][..., None, :])
            self.cache[('sample', num_samples)] = T.concat(samples, -2)
        return self.cache[('sample', num_samples)]

    @classmethod
    def regular_to_natural(cls, l):
        raise NotImplementedError

    @classmethod
    def natural_to_regular(cls, eta):
        raise NotImplementedError

    def log_likelihood(self, x):
        raise NotImplementedError

    @classmethod
    def internal_to_natural(cls, internal_parameters):
        (nh_Q_inv, Q_inv_AB, nh_AB_Q_inv_AB, nh_logdet_Q), prior, q_S, actions, horizon = internal_parameters

        if q_S is not None:
            natparam = Gaussian.unpack(q_S.get_parameters('natural'))
            J, h, log_z = natparam[0], natparam[1], q_S.log_z()
            potentials = [J, h, log_z]
        else:
            potentials = None

        # (A, B, Q), prior, actions = parameters
        # (A, B, Q), prior, potentials, actions = parameters

        ds = T.shape(nh_Q_inv)[-1]
        dsa = T.shape(Q_inv_AB)[-1]

        Q_inv = -2 * nh_Q_inv

        Q_inv_A = Q_inv_AB[..., :ds]
        Q_inv_B = Q_inv_AB[..., ds:]
        logdetQ = -2 * nh_logdet_Q
        A_Q_inv_A = -2 * nh_AB_Q_inv_AB[..., :ds, :ds]
        A_Q_inv_B = -2 * nh_AB_Q_inv_AB[..., :ds, ds:]
        B_Q_inv_B = -2 * nh_AB_Q_inv_AB[..., ds:, ds:]

        H = T.shape(Q_inv_AB)[0] + 1
        N = T.shape(actions)[0]

        Q_inv = T.concatenate([
            Q_inv,
            T.eye(ds)[None]
        ], 0)
        Q_inv_A, Q_inv_B, A_Q_inv_B, A_Q_inv_A, B_Q_inv_B = tuple(map(
            lambda x: T.core.pad(x, [[0, 1], [0, 0], [0, 0]]),
            (Q_inv_A, Q_inv_B, A_Q_inv_B, A_Q_inv_A, B_Q_inv_B)
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
        h1 = -T.einsum('tab,itb->ita', A_Q_inv_B, actions)[..., None]
        h3 =  T.einsum('ita,tab,itb->it', actions, B_Q_inv_B, actions)[..., None]
        J11 = -T.tile(A_Q_inv_A[None], [N, 1, 1, 1])
        J12 = T.tile(t(Q_inv_A)[None], [N, 1, 1, 1])
        J22 = -T.tile(Q_inv[None], [N, 1, 1, 1])
        dynamics_natparam = 0.5 * (
            vs([
                hs([J11,    J12,   h1]),
                hs([t(J12), J22,   h2]),
                hs([t(h1),  t(h2), -h3[..., None] - logdetQ[..., None, None]]),
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
