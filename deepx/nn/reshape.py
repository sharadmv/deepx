import numpy as np

from .. import backend as T

from ..core import Layer, ShapedLayer

class Reshape(ShapedLayer):

    def initialize(self):
        pass

    def infer(self, shape_in):
        return shape_in.copy(dim=self.get_dim_out())

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
        return shape_in.copy(dim=np.product(shape_in.get_dim()))

    def forward(self, X, **kwargs):
        return T.flatten(X)
