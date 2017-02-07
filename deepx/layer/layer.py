from abc import abstractmethod

from .. import T
from ..core import ShapedNode
from ..core import Data
from ..core import Shape
from ..initialization import initialize_weights

class Layer(ShapedNode):

    def __init__(self, initialization=None):
        super(Layer, self).__init__(None, None)
        self.initialized = False
        self.initialization = initialization if initialization is not None else T.get_current_initialization()
        self.parameters = {}

    def get_outputs(self, input, **kwargs):
        raw_input = input.get_placeholder()
        if input.is_sequence():
            raw_output = self.recurrent_forward(raw_input, **kwargs)
        else:
            raw_output = self.forward(raw_input, **kwargs)
        return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    def inputs(self):
        return []

    def forward(self, X):
        pass

    def recurrent_forward(self, X, **kwargs):
        def step(X, _):
            return self.forward(X, **kwargs), []
        outputs, _ = T.rnn(step, X, [])
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

    def create_parameter(self, name, shape, value=None):
        if name not in self.parameters:
            pass
            parameter = T.variable(
                initialize_weights(self.initialization, shape, value=value),
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
        if self.is_elementwise():
            return shape_in
        return shape_in.copy(dim=self.get_shapes_out()[0].get_dim())

    def __str__(self):
        if self.is_elementwise():
            return "%s()" % self.__class__.__name__
        return super(ShapedLayer, self).__str__()

class RecurrentLayer(Layer):

    def __init__(self, stateful=False, **kwargs):
        super(RecurrentLayer, self).__init__(**kwargs)
        self.stateful = stateful

    def forward(self, X, **kwargs):
        if not X.is_sequence():
            raise TypeError("Cannot pass non-sequence into recurrent layer.")
        return self.recurrent_forward(X)

    def recurrent_forward(self, X, **kwargs):
        states = self.get_initial_states(input_data=X)
        def step(X, states):
            return self.step(X, states, **kwargs)
        outputs, states = T.rnn(step, X, states)
        if self.stateful:
            self.updates = zip(self.states, states)
        return outputs

    @abstractmethod
    def step(self, X, states, **kwargs):
        pass

    @abstractmethod
    def create_initial_state(self, input_data, stateful, shape_index=1):
        batch_size = self.get_shapes_in()[0].get_batch_size() or T.shape(input_data)[1]
        dim_out = self.get_dim_out()
        if stateful:
            if not isinstance(batch_size, int):
                raise TypeError("batch_size must be set for stateful RNN.")
            return [T.variable(T.zeros((batch_size, dim_out)))]
        return [T.alloc(0, (batch_size, dim_out), unbroadcast=shape_index)]

    def get_initial_states(self, input_data=None, shape_index=1):
        if self.states is not None:
            return self.states
        states = self.create_initial_state(input_data, self.stateful, shape_index=shape_index)
        if self.stateful:
            self.states = states
        return states

    def reset_states(self):
        if self.states is not None:
            for i, _ in enumerate(self.states):
                self.reset_state(i)

    def reset_state(self, i):
        if self.states is not None:
            T.set_value(self.states[i], T.get_value(self.states[i]) * 0)
