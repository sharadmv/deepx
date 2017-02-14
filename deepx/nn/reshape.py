import numpy as np

from .. import T

from ..layer import Layer, ShapedLayer

class Reshape(ShapedLayer):

    def initialize(self):
        pass

    def infer(self, shape_in):
        dim = shape_in.get_dim()
        first = []
        if shape_in.batch:
            first += dim[:1]
        if shape_in.sequence:
            first += dim[:1]
        return shape_in.copy(dim=first + list(self.dim_out))

    def forward(self, X, **kwargs):
        shape_out = self.get_dim_out()
        if isinstance(shape_out, tuple):
            shape_out = list(shape_out)
        elif not isinstance(shape_out, list):
            shape_out = [shape_out]
        return T.reshape(X, [-1] + shape_out)

class Flatten(Layer):

    def initialize(self):
        pass

    def infer(self, shape_in):
        dim = shape_in.get_dim()
        first = []
        if shape_in.batch:
            first += dim[:1]
            dim = dim[1:]
        if shape_in.sequence:
            first += dim[:1]
            dim = dim[1:]
        return shape_in.copy(dim=first + [np.product(dim)])

    def forward(self, X, **kwargs):
        return T.flatten(X)
