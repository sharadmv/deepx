import numpy as np
from .. import T

from .common import ExponentialFamily
from .gumbel import Gumbel

class Categorical(ExponentialFamily):

    def expected_value(self):
        raise NotImplementedError

    def get_param_dim(self):
        return 1

    def log_h(self, x):
        return T.zeros(T.shape(x)[:-1])

    def sufficient_statistics(self, x):
        return x

    def expected_sufficient_statistics(self):
        return self.get_parameters('regular')

    def sample(self, num_samples=1, temperature=None):
        a = self.get_parameters('natural')
        d = T.shape(a)[-1]
        gumbel_noise = Gumbel(T.zeros_like(a), T.ones_like(a)).sample(num_samples)
        return T.one_hot(T.argmax(a[None] + gumbel_noise, -1), d)

    @classmethod
    def regular_to_natural(cls, pi):
        return T.log(pi)

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        return T.exp(natural_parameters)

    def log_likelihood(self, x):
        a = self.get_parameters('natural')
        return T.sum(a * x, -1)

    def log_z(self):
        p = self.get_parameters('natural')
        return T.logsumexp(p, -1)
