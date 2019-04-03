from deepx.backend import T
from deepx.core import ShapedLayer

class Linear(ShapedLayer):

    def initialize(self):
        dim_in, dim_out = self.get_dim_in()[-1], self.get_dim_out()[-1]
        self.create_parameter('W', [dim_in, dim_out])
        self.create_parameter('b', [dim_out], initial_value=T.zeros([dim_out]))

    def _forward(self, X):
        W, b = self.get_parameter("W"), self.get_parameter("b")
        return T.dot(X, W) + b

FC = Linear
