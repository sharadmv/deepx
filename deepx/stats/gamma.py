from .. import T

from .common import ExponentialFamily

class Gamma(ExponentialFamily):

    def get_param_dim(self):
        return [0, 0]

    def sample(self, num_samples=1):
        a, b = self.get_parameters('regular')
        return T.random_gamma([num_samples], a, b)

    def log_likelihood(self, x):
        a, b = self.get_parameters('regular')
        return (
            a * T.log(b) + (a - 1) * T.log(x) - b * x
            - T.gammaln(a)
        )

    def expected_value(self):
        a, b = self.get_parameters('regular')
        return a / b

    def sufficient_statistics(self, x):
        return [
            T.log(x),
            x
        ]

    def expected_sufficient_statistics(self):
        a, b = self.get_parameters('regular')
        return [
            T.digamma(a) - T.log(b),
            a / b
        ]

    def log_h(self, x):
        d = T.shape(x)[-1]
        return T.zeros([d])

    def log_z(self):
        a, b = self.get_parameters('regular')
        return T.gammaln(a) - a * T.log(b)

    @classmethod
    def natural_to_regular(cls, eta):
        eta1, eta2 = eta
        return [
            eta1 + 1,
            -eta2
        ]

    @classmethod
    def regular_to_natural(cls, regular):
        a, b = regular
        return [
            a - 1,
            -b
        ]
