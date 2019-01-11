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
        return T.log(regular_parameters) - T.log1p(-1. * regular_parameters)

    def natural_to_regular(cls, eta):
        return T.sigmoid(eta)

    def log_likelihood(self, x):
        return -T.sum(T.binary_crossentropy(self.get_parameters('natural'), x, from_logits=True), axis=-1)

    def log_z(self):
        raise NotImplementedError

    def log_h(self, x):
        raise NotImplementedError

    def sufficient_statistics(self, x):
        raise NotImplementedError

    def expected_sufficient_statistics(self):
        raise NotImplementedError
