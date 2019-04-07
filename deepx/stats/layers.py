from deepx.backend import T
from deepx import stats
from deepx.core import ShapedLayer

class Normal(ShapedLayer):

    def initialize(self):
        dim_in, dim_out = self.get_dim_in()[-1], self.get_dim_out()[-1]
        self.create_parameter('W', [dim_in, dim_out * 2])
        self.create_parameter('b', [dim_out * 2], initial_value=T.zeros([dim_out * 2]))

    def _forward(self, X, params=None):
        W, b = self.get_parameter_list("W", "b", params=params)
        logits = T.dot(X, W) + b
        loc, scale = T.split(logits, 2, -1)
        return stats.Normal(loc, T.softplus(scale))

class Gaussian(ShapedLayer):

    def initialize(self):
        dim_in, dim_out = self.get_dim_in()[-1], self.get_dim_out()[-1]
        self.create_parameter('W', [dim_in, dim_out * 2])
        self.create_parameter('b', [dim_out * 2], initial_value=T.zeros([dim_out * 2]))

    def _forward(self, X, params=None):
        W, b = self.get_parameter_list("W", "b", params=params)
        logits = T.dot(X, W) + b
        loc, scale = T.split(logits, 2, -1)
        return stats.GaussianDiag(loc, T.softplus(scale))

class Bernoulli(ShapedLayer):

    def initialize(self):
        dim_in, dim_out = self.get_dim_in()[-1], self.get_dim_out()[-1]
        self.create_parameter('W', [dim_in, dim_out])
        self.create_parameter('b', [dim_out], initial_value=T.zeros([dim_out]))

    def _forward(self, X, params=None):
        W, b = self.get_parameter_list("W", "b", params=params)
        logits = T.dot(X, W) + b
        return stats.Bernoulli(logits=logits)
