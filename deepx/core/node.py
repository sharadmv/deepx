import six
from abc import ABCMeta, abstractmethod

@six.add_metaclass(ABCMeta)
class Node(object):

    def __init__(self):
        self.shapes_in = None
        self.shapes_out = None

    def __call__(self, *args):
        return self.forward(*args)

    @abstractmethod
    def forward(self, *args):
        pass

class Layer(Node):

    def create_parameter(self, name, shape, initial_value=None):
        if name not in self.parameters:
            if initial_value is None:
                parameter = T.variable(
                    initialize_weights(self.initialization, shape),
                    name=name,
                )
            else:
                parameter = T.variable(
                    np.array(initial_value, dtype=T.floatx()),
                    name=name,
                )
            self.parameters[name] = parameter

class ShapedLayer(Layer):

    def __init__(self, shape_in=None, shape_out=None, elementwise=False,
                 sparse=False,
                 **kwargs):
        super(ShapedLayer, self).__init__(**kwargs)
        self.sparse = sparse
        self._elementwise = elementwise
        if shape_out is not None:
            self.dim_in, self.dim_out = shape_in, shape_out
        elif shape_in is not None and shape_out is None:
            self.dim_in, self.dim_out = None, shape_in
        else:
            self._elementwise = True

    def infer_shape(self, X):
        pass

    def forward(self, X):
        pass
