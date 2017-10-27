from .. import T

from .common import ExponentialFamily

class Bernoulli(ExponentialFamily):

    def __init__(self, *args, **kwargs):
        super(Bernoulli, self).__init__(*args, **kwargs)

    def get_param_dim(self):
        return 1

    def expected_value(self):
        return self.get_parameters('regular')

    def sample(self, num_samples=1):
        p = self.get_parameters('regular')
        sample_shape = T.concat([[num_samples], T.shape(p)], 0)
        noise = T.random_uniform(sample_shape)
        sample = T.switch(noise - p[None] < 0, T.ones(sample_shape), T.zeros(sample_shape))
        return sample

    def regular_to_natural(cls, regular_parameters):
        raise NotImplementedError

    def natural_to_regular(cls, natural_parameters):
        raise NotImplementedError

    def log_likelihood(self, x):
        p = self.get_parameters('regular')
        return T.sum(x * T.log(T.epsilon() + p) + (1.0 - x) * T.log(T.epsilon() + 1.0 - p), -1)

    def log_z(self):
        raise NotImplementedError

    def log_h(self, x):
        raise NotImplementedError

    def sufficient_statistics(self, x):
        raise NotImplementedError

    def expected_sufficient_statistics(self):
        raise NotImplementedError
