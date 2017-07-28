from .. import T

from .common import ExponentialFamily

class Categorical(ExponentialFamily):

    def expected_value(self):
        raise NotImplementedError

    def log_h(self, x):
        return T.zeros(T.shape(x)[:-1])

    def sufficient_statistics(self, x):
        return x

    def expected_sufficient_statistics(self):
        return self.get_parameters('regular')

    def sample(self, num_samples=1):
        raise NotImplementedError

    @classmethod
    def regular_to_natural(cls, pi):
        return T.log(pi)

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        return T.exp(natural_parameters)

    def log_likelihood(self, x):
        raise NotImplementedError

    def log_z(self):
        p = self.get_parameters('natural')
        return T.logsumexp(p, -1)
