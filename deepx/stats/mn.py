import numpy as np
from .. import T

from .common import ExponentialFamily

from .gaussian import Gaussian

__all__ = ["MatrixNormal", "MN"]

class MatrixNormal(ExponentialFamily):

    def get_param_dim(self):
        return 2

    def expected_value(self):
        raise NotImplementedError()

    def sample(self, num_samples=1):
        return Gaussian(self.get_parameters('natural'), 'natural').sample(num_samples=num_samples)

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        M, S, V = regular_parameters
        sigma = T.kronecker(V, S)
        mu = T.vec(M)
        return Gaussian.regular_to_natural([sigma, mu])

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        raise NotImplementedError

    def log_likelihood(self, x):
        pass

    def log_z(self):
        eta1, eta2 = Gaussian.unpack(self.get_parameters('natural'))
        eta1_inv = T.matrix_inverse(eta1)
        d = T.to_float(T.shape(eta2)[-1])
        quad_term = T.sum(eta2 * T.sum(eta1_inv * eta2[..., None, :], -1), -1)
        return (d * 0.5 * T.log(2 * np.pi)
                - 0.25 * quad_term
                - 0.5 * T.logdet(-2 * eta1, warn=True))

    def log_h(self, x):
        raise NotImplementedError

    def sufficient_statistics(self, x):
        raise NotImplementedError

    def expected_sufficient_statistics(self):
        eta1, eta2 = Gaussian.unpack(self.get_parameters('natural'))
        sigma = -0.5 * T.matrix_inverse(eta1)
        sigma_inv = -2 * eta1
        mu = T.matrix_solve(sigma_inv, eta2[..., None])[..., 0]
        eta1 = T.outer(mu, mu) + sigma
        eta2 = mu
        return Gaussian.pack([eta1, eta2])

MN = MatrixNormal

