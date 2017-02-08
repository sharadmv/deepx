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

    def get_graph_inputs(self):
        return []

    def get_graph_parameters(self):
        if self.frozen:
            return []
        return self.parameters.values()

    def get_graph_updates(self, **kwargs):
        return self.updates

    def reset_states(self):
        pass

    def reset_state(self, i):
        pass

    # Shape inference

    @abstractmethod
    def infer(self, shape_in):
        pass

    def infer_shape(self):
        if self.initialized:
            return
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            predicted_shapes_out = [self.infer(shapes_in[0])]
            self.set_shapes_out(predicted_shapes_out)
            self.initialize()
            self.initialized = True

    def get_dim_in(self):
        if self.get_shapes_in() is None:
            return None
        shape_in = self.get_shapes_in()[0]
        return shape_in.get_dim()[-1]

    def get_dim_out(self):
        if self.get_shapes_out() is None:
            return None
        shape_out = self.get_shapes_out()[0]
        return shape_out.get_dim()[-1]

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
        return self.parameters

    def get_parameter(self, name):
        return self.parameters[name]

    def get_parameter_list(self, *names):
        return tuple(self.parameters[name] for name in names)

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
                               self.get_dim_in(), self.get_dim_out())

class ShapedLayer(Layer):

    def __init__(self, shape_in=None, shape_out=None, elementwise=False,
                 sparse=False,
                 **kwargs):
        super(ShapedLayer, self).__init__(**kwargs)
        self.sparse = sparse
        self._elementwise = elementwise
        if shape_out is not None:
            pass
        elif shape_in is not None and shape_out is None:
            shape_in, shape_out = None, shape_in
        else:
            self._elementwise = True
        if shape_in is not None:
            self.set_shapes_in([Shape([None, shape_in], batch=True)])
        if shape_out is not None:
            self.set_shapes_out([Shape([None, shape_out], batch=True)])

    def is_elementwise(self):
        return self._elementwise

    def infer(self, shape_in):
        print(shape_in)
        if self.is_elementwise():
            return shape_in
        print(shape_in.copy(dim=self.get_shapes_out()[0].get_dim()))
        return shape_in.copy(dim=self.get_shapes_out()[0].get_dim())

    def __str__(self):
        if self.is_elementwise():
            return "%s()" % self.__class__.__name__
        return super(ShapedLayer, self).__str__()

