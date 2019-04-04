from functools import reduce
from operator import mul

from deepx.core import ShapedLayer, Layer
from deepx.backend import T

class Reshape(ShapedLayer):

    def is_initialized(self):
        return True

    def initialize(self):
        pass

    def _forward(self, X, **kwargs):
        return T.reshape(X, [-1] + self.dim_out)

    def shape_inference(self):
        shape_in = self.get_shape_in()
        dim_in, dim_out = self.get_dim_in(), self.get_dim_out()
        if shape_in is not None:
            assert len(shape_in) == 1, "Cannot pass multiple inputs into layer"
            shape_in = shape_in[0]
            if dim_in is not None:
                if dim_in != shape_in[-len(dim_in):]:
                    raise Exception("Shape inference error")
            self.dim_in = shape_in[1:]
            assert dim_out is not None
        else:
            if dim_in is not None:
                self.set_shape_in([[None] + dim_in])
        self.set_shape_out([self.get_shape_in()[0][:len(self.get_dim_in())] + dim_out])
        if not self.initialized and self.get_shape_in() is not None and self.get_shape_out() is not None:
            self.initialize()
            self.initialized = True

class Flatten(Layer):

    def __init__(self, leading_dim=1):
        super(Flatten, self).__init__()
        self.leading_dim = 1

    def is_initialized(self):
        return self.get_shape_out() is not None

    def initialize(self):
        pass

    def shape_inference(self):
        shape_in = self.get_shape_in()
        if shape_in is not None:
            assert len(shape_in) == 1
            shape_in = shape_in[0]
            out_dim = reduce(mul, shape_in[self.leading_dim:], 1)
            self.set_shape_out([shape_in[:self.leading_dim] + [out_dim]])

    def _forward(self, X):
        return T.reshape(X, self.get_shape_out()[0])

    def __repr__(self):
        shape_in, shape_out = self.get_shape_in(), self.get_shape_out()
        if shape_in is not None:
            shape_in = self.get_shape_in()[0][self.leading_dim:]
        if shape_out is not None:
            shape_out = self.get_shape_out()[0][self.leading_dim:]
        return "Flatten({}, {})".format(
            shape_in, shape_out
        )
