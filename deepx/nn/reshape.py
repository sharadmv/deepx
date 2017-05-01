import numpy as np

from .. import T

from ..layer import Layer

class Reshape(Layer):

    def __init__(self, dim_out, leading=1):
        super(Reshape, self).__init__()
        self.dim_in = None
        self.dim_out = dim_out
        self.leading = leading

    def is_initialized(self):
        return True

    def initialize(self):
        pass

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def infer_shape(self, shape):
        if shape is None: return
        self.dim_in = shape[self.leading - 1:]

    def forward(self, X, **kwargs):
        return T.reshape(X, [-1] * self.leading + self.dim_out)

    def __str__(self):
        return "Reshape(%s, %s)" % (self.dim_in, self.dim_out)

class Flatten(Layer):

    def __init__(self, leading=1):
        super(Flatten, self).__init__()
        self.dim_in = self.dim_out = None
        self.leading = leading

    def is_initialized(self):
        return True

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def initialize(self):
        pass

    def infer_shape(self, shape):
        if shape is None: return
        self.dim_in = shape[self.leading - 1:]
        self.dim_out = [np.prod(self.dim_in)]

    def forward(self, X, **kwargs):
        leading_shape = T.get_shape(X)[:self.leading]
        result = T.flatten(X, leading=self.leading)
        result.set_shape(leading_shape + self.dim_out)
        return result

    def __str__(self):
        return "Flatten(%s)" % self.dim_out
