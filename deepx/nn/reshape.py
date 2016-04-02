import numpy as np

from .. import backend as T

from ..core import Layer, ShapedLayer

class Reshape(ShapedLayer):

    def initialize(self):
        pass

    def _infer(self, shape_in):
        return self.get_shape_out()

    def _forward(self, X):
        shape_out = self.get_shape_out()
        if isinstance(shape_out, tuple):
            shape_out = list(shape_out)
        elif not isinstance(shape_out, list):
            shape_out = [shape_out]
        return T.reshape(X, [-1] + shape_out)

class Flatten(Layer):

    def initialize(self):
        pass

    def _infer(self, shape_in):
        return np.product(shape_in)

    def _forward(self, X):
        return T.flatten(X)

    def to_str(self):
        return "Flatten(%s, %s)" % (self.get_shape_in(), self.get_shape_out())
