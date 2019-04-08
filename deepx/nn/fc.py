from deepx.backend import T
from deepx.core import ShapedLayer

class Linear(ShapedLayer):
    """A layer that performs matrix multiply and bias add.
    """

    def initialize(self):
        dim_in, dim_out = self.get_dim_in()[-1], self.get_dim_out()[-1]
        self.create_parameter('W', [dim_in, dim_out])
        self.create_parameter('b', [dim_out], initial_value=T.zeros([dim_out]))

    def _forward(self, X, params=None):
        W, b = self.get_parameter_list("W", "b", params=params)
        return T.dot(X, W) + b

FC = Linear
