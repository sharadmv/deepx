from .. import T

from .common import ExponentialFamily

class Dirichlet(ExponentialFamily):

    def sample(self, num_samples=1):
        raise NotImplementedError

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
        return T.reduce_sum(T.gammaln(alpha), -1) - T.gammaln(T.sum(alpha, -1))

    @classmethod
    def natural_to_regular(cls, eta):
        return eta + 1

    @classmethod
    def regular_to_natural(cls, alpha):
        return alpha - 1
