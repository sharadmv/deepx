from .. import T

from .common import ExponentialFamily

class Exponential(ExponentialFamily):

    def get_param_dim(self):
        return 1

    def expected_value(self):
        return 1 / self.get_parameters('regular')

    def log_h(self, x):
        return T.zeros(T.shape(x)[:-1])

    def sufficient_statistics(self, x):
        return x

    def expected_sufficient_statistics(self):
        return 1 / self.get_parameters('regular')

    def sample(self, num_samples=1):
        l = self.get_parameters('regular')
        sample_shape = T.concat([[num_samples], T.shape(l)])
        noise = T.random_uniform(sample_shape)
        return -T.log(noise) / l[None]

    @classmethod
    def regular_to_natural(cls, l):
        return -l

    @classmethod
    def natural_to_regular(cls, eta):
        return -eta

    def log_likelihood(self, x):
        l = self.get_parameters('regular')
        return T.sum(T.log(l) - x * l,  T.range(-T.rank(l), 0, 1))

    def log_z(self):
        l = self.get_parameters('regular')
        return -T.log(l)
