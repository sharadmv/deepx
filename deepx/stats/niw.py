import numpy as np

from .. import T

from .common import ExponentialFamily
from .iw import IW
from .gaussian import Gaussian
from . import util

__all__ = ["NormalInverseWishart", "NIW"]

class NormalInverseWishart(ExponentialFamily):

    def get_param_dim(self):
        return 2

    def sample(self, num_samples=1):
        S, mu0, kappa, nu = self.get_parameters('regular')
        sigma = IW([S, nu]).sample(num_samples=num_samples)
        tiled_mu0 = T.tile((1 / kappa * mu0)[None], T.concatenate([[num_samples], T.cast(T.ones(T.rank(mu0)), T.int32)]))
        mu = Gaussian([sigma, tiled_mu0]).sample()[0]
        return [sigma, mu]

    def log_likelihood(self, x):
        stats = self.sufficient_statistics(x)
        params = self.get_parameters('natural')
        log_h = self.log_h(x)
        log_z = self.log_z()
        exponent_term = T.sum(stats * params, [-1, -2])
        return log_h + exponent_term - log_z

    def expected_value(self):
        S, mu0, kappa, nu = self.get_parameters('regular')
        e_sigma = IW([S, nu]).expected_value()
        return e_sigma, mu0

    def sufficient_statistics(self, x):
        sigma, mu = x
        sigma_inv = T.matrix_inverse(sigma)
        h = T.matrix_solve(sigma, mu[..., None])[..., 0]
        return self.pack([
            -1 / 2. * sigma_inv,                 # -\frac{1}{2} \Sigma^{-1}
            h,                                   # \Sigma^{-1} \mu
            -1 / 2. * T.sum(mu * h, -1), # -\frac{1}{2} \mu^T \Sigma^{-1} \mu
            -1 / 2. * T.logdet(sigma)              # -\frac{1}{2} \log |\Sigma|
        ])


    def expected_sufficient_statistics(self, fudge=1e-8):
        S, mu0, kappa, nu = self.get_parameters('regular')
        d = T.shape(mu0)[-1]
        S_inv = T.matrix_inverse(S)
        g1 = -0.5 * nu[..., None, None] * S_inv
        g2 = nu[..., None] * T.matmul(S_inv, mu0[..., None])[..., 0]
        g3 = -0.5 * (T.matmul(mu0[..., None, :], g2[..., None]) + T.cast(d, T.dtype(S)) / kappa[..., None])[..., 0, 0]
        g4 = 0.5 * (T.cast(d, T.dtype(S)) * T.cast(T.log(2.), T.dtype(S))
                     - T.logdet(S)
                     + T.sum(T.digamma((nu[...,None] - T.cast(T.range(d)[None,...], T.dtype(S)))/2.), -1))
        return self.pack([g1, g2, g3, g4])

    def log_h(self, x):
        sigma, mu = x
        d = T.to_float(T.shape(mu)[-1])
        return (-d / 2. - 1) * T.logdet(sigma) - d / 2. * T.log(2 * np.pi)

    def log_z(self):
        S, mu0, kappa, nu = self.get_parameters('regular')
        d = T.to_float(T.shape(mu0)[-1])
        return (nu / 2. * (d * T.log(2.) - T.logdet(S))
                + T.multigammaln(nu / 2., d)
                - d / 2. * T.log(kappa + 1e-6))

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        S, mu0, kappa, nu = regular_parameters
        d = T.shape(mu0)[-1]
        b = T.expand_dims(kappa, -1) * mu0
        A = S + T.outer(b, mu0)
        return cls.pack([A, b, kappa, nu + T.to_float(d) + 1])

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        A, b, kappa, nu = cls.unpack(natural_parameters)
        m = b / T.expand_dims(kappa, -1)
        S = A - T.outer(b, m)
        return [S, m, kappa, nu]

    @classmethod
    def pack(cls, parameters):
        return util.pack(parameters)

    @classmethod
    def unpack(cls, parameters):
        return util.unpack(parameters)
NIW = NormalInverseWishart
