import numpy as np
from .. import backend as T
from .exceptions import ShapeOutError
from .node import Node
from .shape import Shape
from .initialization import initialize_weights

class Layer(Node):

    def __init__(self, weight_init='uniform'):
        super(Layer, self).__init__()
        self.weight_init = weight_init
        self.parameters = {}

    def get_inputs(self):
        return []

    def can_initialize(self):
        return self.get_shape_in() is not None and self.get_shape_out() is not None

    def get_dim_in(self):
        if self.get_shape_in() is None:
            return None
        shape_in = self.get_shape_in()[0].get_dim()
        if len(shape_in) == 1:
            return shape_in[0]
        return shape_in

    def get_dim_out(self):
        if self.get_shape_out() is None:
            return None
        shape_out = self.get_shape_out()[0].get_dim()
        if len(shape_out) == 1:
            return shape_out[0]
        return shape_out

    def reinitialize(self):
        self.parameters = {}
        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def forward(self, inputs, **kwargs):
        from .data import Data
        if inputs[0].is_sequence():
            return self.recurrent_forward(inputs, **kwargs)
        raw_inputs = [data.get_placeholder() for data in inputs]
        raw_outputs = self._forward(*raw_inputs, **kwargs)
        if not (isinstance(raw_outputs, tuple) or isinstance(raw_outputs, list)):
            raw_outputs = [raw_outputs]
        outputs = []
        for raw_output, shape_out in zip(raw_outputs, self.get_shape_out()):
            outputs.append(Data(shape_out, placeholder=raw_output))
        return outputs

    def get_initial_states(self, *args, **kwargs):
        return []

    def recurrent_forward(self, inputs, **kwargs):
        from .data import Data
        seq = inputs[0]
        non_seqs = inputs[1:]
        output = self._recurrent_forward(seq.get_placeholder(), [x.get_placeholder() for x in non_seqs])
        return [Data(self.get_shape_out()[0], placeholder=output)]

    def _recurrent_forward(self, input_seq, non_seqs):

        input_states = self.get_initial_states(input_data=input_seq)

        def step(input, states, non_seqs):
            return self._forward(input, *(states + non_seqs)), []

        output = T.rnn(step, input_seq, input_states, non_sequences=non_seqs)[0]
        return output

    def init_parameter(self, name, shape, value=None):
        if name not in self.parameters:
            parameter = T.variable(
                initialize_weights(shape, self.weight_init, value=value),
                name=name,
            )
            self.parameters[name] = parameter

    def infer_shape(self):
        shape_in = self.get_shape_in()
        shape_out = self.get_shape_out()
        if shape_in is not None:
            predicted_shape_out = [self._infer(shape_in[0])]
            if shape_out is None:
                self.set_shape_out(predicted_shape_out)
            elif shape_out != predicted_shape_out:
                raise ShapeOutError(self, shape_out)

    def get_graph_parameters(self):
        if self.frozen:
            return []
        return self.parameters.values()

    def get_parameters(self):
        return self.parameters

    def get_parameter(self, name):
        return self.parameters[name]

    def get_parameter_value(self, name):
        return T.get_value(self.get_parameter(name))

    def set_parameter_value(self, name, value):
        T.set_value(self.get_parameter(name), value)

    def get_parameter_value_list(self, *names):
        values = []
        for name in names:
            values.append(self.get_parameter_value(name))
        return values

    def get_parameter_tree(self):
        return self.parameters

    def set_parameter_tree(self, params):
        for key, val in params.items():
            self.parameters[key] = val

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

    def is_recurrent(self):
        return False

    def tie(self, node):
        new_node = self.copy(keep_params=True)
        for key, val in node.parameters.items():
            new_node.parameters[key] = val
        return new_node

    def __str__(self):
        shape_in = self.get_shape_in()
        if shape_in is not None:
            shape_in = shape_in[0]
        shape_out = self.get_shape_out()
        if shape_out is not None:
            shape_out = shape_out[0]
        return "%s(%s, %s)" % (self.__class__.__name__,
                               shape_in, shape_out)

class ShapedLayer(Layer):

    def __init__(self, shape_in=None, shape_out=None, elementwise=False,
                 **kwargs):
        super(ShapedLayer, self).__init__(**kwargs)
        self._elementwise = elementwise
        if shape_out is not None:
            shape_in, shape_out = (shape_in, shape_out)
        elif shape_in is not None and shape_out is None:
            shape_in, shape_out = None, shape_in
        else:
            self._elementwise = True
        if shape_in is not None:
            self.set_shape_in([Shape(shape_in)])
        if shape_out is not None:
            self.set_shape_out([Shape(shape_out)])

    def is_elementwise(self):
        return self._elementwise

    def _infer(self, shape_in):
        if self.is_elementwise():
            return shape_in
        return self.get_shape_out()[0]

    def __str__(self):
        if self.is_elementwise():
            return "%s()" % self.__class__.__name__
        return super(ShapedLayer, self).__str__()

class RecurrentLayer(Layer):

    def __init__(self, stateful=False):
        super(RecurrentLayer, self).__init__()
        self.stateful = stateful

    def get_initial_states(self, input_data=None, shape_index=1):
        batch_size = self.get_shape_in()[0].get_batch_size() or T.shape(input_data.get_placeholder())[0]
        dim_out = self.get_dim_out()
        if self.stateful:
            if not isinstance(batch_size, int):
                raise TypeError("batch_size must be set for stateful RNN.")
            return [T.variable(np.zeros((batch_size, dim_out)))]
        return [T.alloc(0, (batch_size, dim_out), unbroadcast=shape_index)]

    def reset_states(self):
        if self.states is not None:
            for i, _ in enumerate(self.states):
                self.reset_state(i)

    def reset_state(self, i):
        if self.states is not None:
            T.set_value(self.states[i], T.get_value(self.states[i]) * 0)

    def recurrent_forward(self, inputs, **kwargs):
        from .data import Data
        raw_inputs = [x.get_placeholder() for x in inputs]
        raw_output = self._forward(*raw_inputs)
        return [Data(self.get_shape_out()[0],
                      placeholder=raw_output)]

    def step(self, X, state):
        out, state = self._step(X.get_placeholder(), state, [])
        return Data.from_placeholder(out, self.get_shape_out(), X.batch_size), state

    def _forward(self, X, **kwargs):

        if self.stateful:
            if self.states is None:
                self.states = self.get_initial_states(input_data=X)
            states = self.states
        else:
            states = self.get_initial_states(input_data=X)

        output, new_state = T.rnn(self._step,
                              X,
                              states)
        if self.stateful:
            self.updates = []
            for state, ns in zip(self.states, new_state):
                if (state, ns) not in self.updates:
                    self.updates.append((state, ns))
        return output

    def is_recurrent(self):
        return True
