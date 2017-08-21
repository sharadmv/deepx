import numpy as np
from .. import T

from .common import Distribution

class Gumbel(Distribution):

    def __init__(self, m, b):
        self.m, self.b = m, b

    def sample(self, num_samples=1):
        sample_shape = T.concat([[num_samples], T.shape(self.m)])
        noise = T.random_uniform(sample_shape)
        m, b = self.m[None], self.b[None]
        return m - b * T.log(-T.log(noise))

    def expected_value(self):
        return self.m + np.euler_gamma * self.b

    def log_likelihood(self, x):
        z = (x - self.m) / self.b
        return 1 / self.b * T.exp(-z + T.exp(-z))
