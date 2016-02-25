import numpy as np

from .. import backend as T

from ..node import Node

class Reshape(Node):

    def __init__(self, shape_in, shape_out=None):
        super(Reshape, self).__init__()
        if shape_out is None:
            self.shape_in = None
            self.shape_out = shape_in
        else:
            self.shape_in = shape_in
            self.shape_out = shape_out

    def can_initialize(self):
        return True

    def _infer(self, shape_in):
        return self.shape_out

    def _forward(self, X):
        shape_out = self.get_shape_out()
        if isinstance(shape_out, tuple):
            shape_out = list(shape_out)
        elif not isinstance(shape_out, list):
            shape_out = [shape_out]
        return T.reshape(X, [-1] + shape_out)

class Flatten(Node):

    def can_initialize(self):
        return True

    def _infer(self, shape_in):
        return np.product(shape_in)

    def _forward(self, X):
        return T.flatten(X)

    def to_str(self):
        return "Flatten(%s, %s)" % (self.get_shape_in(), self.get_shape_out())
