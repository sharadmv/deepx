from .. import T

from .common import ExponentialFamily

class Dirichlet(ExponentialFamily):

    def get_param_dim(self):
        return 1

    def sample(self, num_samples=1):
        a = self.get_parameters('regular')
        if isinstance(num_samples, int):
            num_samples = [num_samples]
        sample_shape = T.concat([num_samples, T.shape(a)[:-1]], 0)
        gamma_samples = T.random_gamma(sample_shape, a, T.ones_like(a))
        return gamma_samples / T.sum(gamma_samples, -1)[..., None]

    def log_likelihood(self, x):
        raise NotImplementedError

    def expected_value(self):
        alpha = self.get_parameters('regular')
        return alpha / T.sum(alpha, -1)[..., None]

    def sufficient_statistics(self, x):
        return T.log(x)

    def expected_sufficient_statistics(self):
        alpha = self.get_parameters('regular')
        return T.digamma(alpha) - T.digamma(T.sum(alpha, -1))[..., None]

    def log_h(self, x):
        d = T.shape(x)[-1]
        return T.zeros([d])

    def log_z(self):
        alpha = self.get_parameters('regular')
        return T.sum(T.gammaln(alpha), -1) - T.gammaln(T.sum(alpha, -1))

    @classmethod
    def natural_to_regular(cls, eta):
        return eta + 1

    @classmethod
    def regular_to_natural(cls, alpha):
        return alpha - 1
