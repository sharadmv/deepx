from ..core import Node
from .. import T

__all__ = ["Layer", "ShapedLayer"]

class Layer(Node):
    pass

class ShapedLayer(Layer):

    def __init__(self, shape_in=None, shape_out=None, elementwise=False,
                 sparse=False,
                 **kwargs):
        super(ShapedLayer, self).__init__(**kwargs)
        self.sparse = sparse
        self.elementwise = elementwise
        if shape_out is not None:
            if not isinstance(shape_in, list):
                shape_in = [shape_in]
            if not isinstance(shape_out, list):
                shape_out = [shape_out]
            self.dim_in, self.dim_out = shape_in, shape_out
        elif shape_in is not None and shape_out is None:
            if not isinstance(shape_in, list):
                shape_in = [shape_in]
            self.dim_in, self.dim_out = None, shape_in
        else:
            self.dim_in = self.dim_out = None
            self.elementwise = True

    def is_initialized(self):
        return self.elementwise or not (self.dim_in is None or self.dim_out is None)

    def get_dim_in(self):
        return self.dim_in[0]

    def get_dim_out(self):
        return self.dim_out[0]

    def infer_shape(self, X):
        shape = T.get_shape(X)
        if self.dim_in is None:
            self.dim_in = shape[-1:]
        if self.dim_out is None and self.elementwise:
            self.dim_out = self.dim_in

    def __str__(self):
        name = self.__class__.__name__
        if self.elementwise:
            return "%s()" % name
        return "%s(%s, %s)" % (name, self.dim_in, self.dim_out)
