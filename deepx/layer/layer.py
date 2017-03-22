import numpy as np
from abc import abstractmethod

from .. import T
from ..core import ShapedNode
from ..core import Shape
from ..initialization import initialize_weights

__all__ = ["Layer", "ShapedLayer"]

class Layer(ShapedNode):

    def __init__(self, initialization=None):
        super(Layer, self).__init__(None, None)
        self.initialized = False
        self.initialization = initialization if initialization is not None else T.get_current_initialization()
        self.parameters = {}

    def inputs(self):
        return []

    def outputs(self, X):
        shape_in = self.get_shapes_in()[0]
        if shape_in.sequence:
            return [self.recurrent_forward(X)]
        return [self.forward(X)]

    @abstractmethod
    def forward(self, X):
        pass

    def recurrent_forward(self, X, **kwargs):
        def step(x):
            return self.forward(x, **kwargs)
        outputs = T.map(step, X)
        return outputs

    # Shape inference

    @abstractmethod
    def infer(self, shape_in):
        pass

    def infer_shape(self):
        if self.initialized:
            return
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            self.set_shapes_out([self.infer(shapes_in[0])])
            self.initialize()
            self.initialized = True

    def get_shape_in(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            shapes_in = shapes_in[0]
        return shapes_in

    def get_shape_out(self):
        shapes_out = self.get_shapes_out()
        if shapes_out is not None:
            shapes_out = shapes_out[0]
        return shapes_out

    def set_shape_in(self, shape_in):
        self.set_shapes_in([shape_in])

    def set_shape_out(self, shape_out):
        self.set_shapes_out([shape_out])

    def create_parameter(self, name, shape, initial_value=None):
        if name not in self.parameters:
            if initial_value is None:
                parameter = T.variable(
                    initialize_weights(self.initialization, shape),
                    name=name,
                )
            else:
                parameter = T.variable(
                    np.array(initial_value).astype(T.floatx()),
                    name=name,
                )
            self.parameters[name] = parameter

    def get_parameters(self):
        return list(self.parameters.values())

    def get_parameter(self, name):
        return self.parameters[name]

    def get_parameter_list(self, *names):
        return list(self.parameters[name] for name in names)

    def get_parameter_value(self, name):
        return T.get_value(self.get_parameter(name))

    def set_parameter_value(self, name, value):
        T.set_value(self.get_parameter(name), value)

    def get_state(self, as_list=False):
        if as_list:
            return {
                k: T.get_value(v).tolist() for k, v in self.parameters.items()
            }
        return {
            k: T.get_value(v) for k, v in self.parameters.items()
        }

    def set_state(self, state):
        for k, v in state.items():
            self.set_parameter_value(k, v)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.get_shape_in(), self.get_shape_out())

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
        if not self.is_elementwise():
            if self.dim_in is not None:
                self.set_shape_in(Shape([None, self.dim_in], batch=True))
            if self.dim_out is not None:
                self.set_shape_out(Shape([None, self.dim_out], batch=True))

    def get_dim_in(self):
        return self.get_shape_in().get_dim()[-1]

    def get_dim_out(self):
        return self.get_shape_out().get_dim()[-1]

    def is_elementwise(self):
        return self._elementwise

    def infer(self, shape_in):
        if self.is_elementwise():
            return shape_in
        return shape_in.copy(shape=shape_in.get_shape()[:-1] + [self.dim_out])

    def __str__(self):
        if self.is_elementwise():
            return "%s()" % self.__class__.__name__
        return "%s(%u, %u)" % (self.__class__.__name__, self.get_dim_in(), self.get_dim_out())
