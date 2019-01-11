from .. import T
import numpy as np

from .gaussian import Gaussian
from . import util

class GaussianScaleDiag(Gaussian):

    def expected_value(self):
        _, mu = self.get_parameters('regular')
        return mu

    def log_likelihood(self, x):
        scale_diag, mu = self.get_parameters('regular')
        d = T.to_float(T.shape(mu)[0])
        z_score = (x - mu) / scale_diag
        exp_term = T.sum(z_score ** 2, -1)
        return -0.5 * d * T.log(2 * np.pi) - T.sum(T.log(scale_diag), axis=-1) - 0.5 * exp_term

    def log_h(self, x):
        d = T.to_float(T.shape(x)[-1])
        return 0.5 * d * T.log(2 * np.pi)

    def sufficient_statistics(self, x):
        return self.pack([
            T.outer(x, x),
            x,
        ])

    def expected_sufficient_statistics(self):
        scale_diag, mu = self.get_parameters('regular')
        eta1 = T.outer(mu, mu) + T.square(T.matrix_diag(scale_diag))
        eta2 = mu
        return self.pack([eta1, eta2])

    def sample(self, num_samples=1):
        scale_diag, mu = self.get_parameters('regular')
        sample_shape = T.concat([[num_samples], T.shape(mu)], 0)
        noise = T.random_normal(sample_shape)
        return mu[None] + scale_diag[None] * noise

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        scale_diag, mu = regular_parameters
        sigma_inv_diag = T.pow(scale_diag, -2)
        eta2 = mu * sigma_inv_diag
        eta1 = -0.5 * T.matrix_diag(sigma_inv_diag) # -\frac{1}{2} \Sigma^{-1}
        return Gaussian.pack([eta1, eta2])

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        J, h = Gaussian.unpack(natural_parameters)
        sigma_inv = -2 * J
        sigma_inv_diag = T.matrix_diag_part(sigma_inv)
        mu = h / sigma_inv_diag
        return [T.sqrt(1 / sigma_inv_diag), mu]

    @classmethod
    def pack(cls, natural_parameters):
        eta1, eta2 = natural_parameters
        d = T.shape(eta2)[:-1]
        eta3 = eta4 = T.zeros(d)
        return util.pack([eta1, eta2, eta3, eta4])

    @classmethod
    def unpack(cls, packed_parameters):
        eta1, eta2, _, _ = util.unpack(packed_parameters)
        return [eta1, eta2]

    def log_z(self):
        scale_diag, mu = self.get_parameters('regular')
        return 0.5 * T.sum(T.square(mu / scale_diag), -1) + T.sum(T.log(scale_diag), axis=-1)

    def to_tfp(self):
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        scale_diag, mu = self.get_parameters('regular')
        return tfd.MultivariateNormalDiag(
            loc=mu,
            scale_diag=scale_diag
        )
