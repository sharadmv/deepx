from .. import T
from ..core import Layer

class BatchNorm(Layer):

    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def initialize(self):
        self.create_parameter('beta', self.get_dim_in(), value=0)
        self.create_parameter('gamma', self.get_dim_in(), value=1)
        self.

    def infer(self, shape_in):
        return shape_in

    def forward(self, X):
        beta, gamma = self.get_parameter('beta'), self.get_parameter('gamma')
        return T.batch_norm(X, beta, gamma)
