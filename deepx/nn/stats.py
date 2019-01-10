import numpy as np
from .. import T
from ..layer import ShapedLayer
from ..initialization import initialize_weights
from .full import Linear

from .. import stats


__all__ = ['Gaussian', 'Bernoulli', 'IdentityVariance']

class Gaussian(Linear):

    def initialize(self):
        if not self.elementwise:
            dim_in, dim_out = self.get_dim_in()[-1], self.get_dim_out()[-1]
            left = initialize_weights(self.initialization, [dim_in, dim_out // 2])
            right = T.zeros([dim_in, dim_out // 2])
            self.create_parameter('W', [dim_in, dim_out], initial_value=(
                T.concatenate([
                    left, right
                ], -1)
            ))
            self.create_parameter('b', [dim_out], initial_value=np.zeros([dim_out]))

    def __init__(self, *args, **kwargs):
        self.cov_type = kwargs.pop('cov_type', 'diagonal')
        super(Gaussian, self).__init__(*args, **kwargs)
        assert not self.elementwise

    def get_dim_out(self):
        return [self.dim_out[0] * 2]

    def activate(self, X):
        if self.cov_type == 'diagonal':
            log_sigma, mu = T.split(X, 2, axis=-1)
            sigma = T.matrix_diag(T.softplus(log_sigma) + 1e-5)
            return stats.Gaussian([sigma, mu], parameter_type='regular')
        raise Exception("Undefined covariance type: %s" % self.cov_type)

    def __str__(self):
        return "Gaussian(%s)" % self.dim_out


class Bernoulli(Linear):

    def __init__(self, *args, **kwargs):
        self.parameter_type = kwargs.pop('parameter_type', 'natural')
        super(Bernoulli, self).__init__(*args, **kwargs)

    def activate(self, X):
        if self.elementwise:
            return stats.Bernoulli(X, parameter_type=self.parameter_type)
        return stats.Bernoulli(X, parameter_type=self.parameter_type)

    def __str__(self):
        return "Bernoulli(%s)" % self.dim_out


class IdentityVariance(ShapedLayer):

    def __init__(self, variance=1e-4, *args, **kwargs):
        self.variance = variance
        super(IdentityVariance, self).__init__(*args, **kwargs)

    def initialize(self):
        pass

    def get_parameters(self):
        return []

    def infer_shape(self, shape):
        if shape is None: return
        if self.elementwise:
            self.dim_in = shape
            self.dim_out = shape
            return
        if self.dim_in is None:
            self.dim_in = shape

    def forward(self, X):
        shape = T.shape(X)
        batch_shape = shape[:-1]
        dim_out = shape[-1]
        return stats.Gaussian([self.variance * T.eye(dim_out, batch_shape=batch_shape), X])
