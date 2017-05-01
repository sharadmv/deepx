import numpy as np
from .. import T

from ..layer import ShapedLayer

class Linear(ShapedLayer):

    def initialize(self):
        if not self.elementwise:
            dim_in, dim_out = self.get_dim_in()[0], self.get_dim_out()[0]
            self.create_parameter('W', [dim_in, dim_out])
            self.create_parameter('b', [dim_out], initial_value=np.zeros([dim_out]))

    def activate(self, X):
        if self.elementwise:
            raise Exception("No identity nodes")
        return X

    def forward(self, X, **kwargs):
        if self.elementwise:
            return self.activate(X)
        W, b = self.get_parameter_list('W', 'b')
        if self.sparse:
            return self.activate(T.sparse_dot(X, W) + b)
        return self.activate(T.dot(X, W) + b)

class Maxout(Linear):


    def __init__(self, shape_in=None, shape_out=None, k=4, **kwargs):
        super(Maxout, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.k = k

    def initialize(self):
        dim_in, dim_out = self.get_dim_in()[0], self.get_dim_out()[0]
        self.create_parameter('W', (self.k, dim_in, dim_out))
        self.create_parameter('b', (self.k, dim_out))

    def activate(self, X):
        return T.max(X, axis=1)

class Softmax(Linear):

    def __init__(self, shape_in=None, shape_out=None, temp=1.0, temperature_parameter=False,
                 **kwargs):
        super(Softmax, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.T = temp
        self.temperature_parameter = temperature_parameter
        if self.temperature_parameter:
            self.T = T.placeholder(ndim=0, name='temperature')

    def get_graph_inputs(self):
        if self.temperature_parameter:
            return [self.T]
        return []

    def activate(self, X):
        return T.softmax(X, self.T)

class Sigmoid(Linear):

    def activate(self, X):
        return T.sigmoid(X)

class Tanh(Linear):

    def activate(self, X):
        return T.tanh(X)

class Relu(Linear):

    def activate(self, X):
        return T.relu(X)

class Elu(Linear):

    def __init__(self, shape_in=None, shape_out=None, alpha=1.0, **kwargs):
        super(Elu, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.alpha = alpha

    def activate(self, X):
        return T.relu(X) + self.alpha * (T.exp((X - abs(X)) * 0.5) - 1)

class LeakyRelu(Linear):

    def __init__(self, shape_in=None, shape_out=None, alpha=1.0, **kwargs):
        super(LeakyRelu, self).__init__(shape_in=shape_in,
                                     shape_out=shape_out,
                                     **kwargs)
        self.alpha = alpha

    def activate(self, X):
        return T.relu(X, alpha=self.alpha)
