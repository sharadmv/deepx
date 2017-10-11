import numpy as np
from .. import T

from .common import Distribution

class Gumbel(Distribution):

    def get_param_dim(self):
        return 1

    def __init__(self, m, b):
        if isinstance(m, int) or isinstance(m, float):
            m = T.to_float(T.constant(m))
        if isinstance(b, int) or isinstance(b, float):
            b = T.to_float(T.constant(b))
        self.m, self.b = m, b

    def expected_value(self):
        return self.m + np.euler_gamma * self.b

    def log_likelihood(self, x):
        z = (x - self.m) / self.b
        return 1 / self.b * T.exp(-z - T.exp(-z))

    def sample(self, num_samples=1):
        shape = T.shape(self.m)
        sample_shape = T.concat([[num_samples], shape], 0)
        random_sample = T.random_uniform(sample_shape)
        return self.m[None] - self.b[None] * T.log(-T.log(random_sample))
