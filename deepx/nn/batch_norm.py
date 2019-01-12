import numpy as np
from .. import T
from ..layer import Layer

class BatchNorm(Layer):

    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def initialize(self):
        self.create_parameter('beta', self.get_dim_in(), initial_value=np.zeros(self.get_dim_in()))
        self.create_parameter('gamma', self.get_dim_in(), initial_value=np.ones(self.get_dim_out()))

    def infer_shape(self, shape):
        if shape is None: return
        self.dim_in = shape
        self.dim_out = shape

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def is_initialized(self):
        return not (self.dim_in is None or self.dim_out is None)

    def forward(self, X):
        beta, gamma = self.get_parameter('beta'), self.get_parameter('gamma')
        return T.batch_norm(X, beta, gamma)

    def __str__(self):
        name = self.__class__.__name__
        return "%s()" % (name)
