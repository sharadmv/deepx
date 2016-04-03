from .. import backend as T
from .node import ShapedNode
from .data import Data
from .initialization import initialize_weights

class Layer(ShapedNode):

    def __init__(self, *args, **kwargs):
        self.weight_init = kwargs.pop('weight_init', 'default')
        self.parameters = {}
        self.initialized = False
        self.config = kwargs

        super(Layer, self).__init__(*args, shape_in=kwargs.get('shape_in', None),
                                    shape_out=kwargs.get('shape_out', None))

    def forward(self, *inputs, **kwargs):

        sequence = any(x.is_sequence() for x in inputs)
        ignore_sequence = kwargs.pop('ignore_sequence', False)

        if not ignore_sequence and sequence:
            return self.recurrent_forward(*inputs, **kwargs)

        use_kwargs = kwargs.pop('use_kwargs', False)

        raw_inputs = [data.get_placeholder() for data in inputs]
        batch_size = inputs[0].batch_size

        if use_kwargs:
            raw_outputs = self._forward(*raw_inputs, **kwargs)
        else:
            raw_outputs = self._forward(*raw_inputs)

        if not (isinstance(raw_outputs, tuple) or isinstance(raw_outputs, list)):
            raw_outputs = [raw_outputs]

        outputs = []
        shape_out = self.get_shape_out()
        if not isinstance(shape_out, list):
            shape_out = [shape_out]
        for raw_output, dim in zip(raw_outputs, shape_out):
            outputs.append(Data.from_placeholder(raw_output, dim, batch_size, sequence=sequence))
        return outputs

    def recurrent_forward(self, *inputs, **kwargs):
        batch_size = inputs[0].batch_size
        seqs = [x for x in inputs if x.is_sequence()]
        non_seqs = [x for x in inputs if not x.is_sequence()]
        raw_seqs = [seq.get_placeholder() for seq in seqs]
        raw_non_seqs = [x.get_placeholder() for x in non_seqs]
        output = self._recurrent_forward(raw_seqs, raw_non_seqs)
        return [Data.from_placeholder(output, self.get_shape_out(),
                                     batch_size, sequence=True)]

    def _recurrent_forward(self, input_seqs, non_seqs):

        def step(inputs, states, non_seqs):
            return self._forward(*(inputs + states + non_seqs)), []

        output = T.rnn(step, input_seqs, [], non_sequences=non_seqs)[1]
        return output

    def get_config(self):
        return self.config

    def init_parameter(self, name, shape, value=None):
        parameter = T.variable(
            initialize_weights(shape, self.weight_init, value=value),
            name=name,
        )
        self.parameters[name] = parameter

    def initialize(self):
        raise NotImplementedError

    def infer_shape(self):
        super(Layer, self).infer_shape()
        if self.is_configured():
            if not self.initialized:
                self.initialize()
                self.initialized = True

    def has_parameters(self):
        return True

    def get_parameters(self):
        if self.frozen:
            return []
        return self.parameters.values()

    def get_parameter(self, name):
        return self.parameters[name]

    def get_parameter_list(self, *names):
        params = []
        for name in names:
            params.append(self.get_parameter(name))
        return params

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

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, str(self.get_shape_in()), str(self.get_shape_out()))

    def copy(self, keep_params=False):
        kwargs = self.get_config()
        node = self.__class__(**kwargs)
        if keep_params:
            old_params = self.get_parameter_tree()
            node.set_parameter_tree(old_params)
            if self.initialized:
                node.initialized = True
        node.infer_shape()
        return node

    def is_recurrent(self):
        return False

class ShapedLayer(Layer):

    def __init__(self, shape_in=None, shape_out=None, elementwise=False,
                 **kwargs):

        self._elementwise = elementwise
        if shape_out is not None:
            shape_in, shape_out = (shape_in, shape_out)
        elif shape_in is not None and shape_out is None:
            shape_in, shape_out = None, shape_in
        else:
            self._elementwise = True

        super(ShapedLayer, self).__init__(shape_in=shape_in, shape_out=shape_out, **kwargs)

    def is_elementwise(self):
        return self._elementwise

    def get_config(self):
        config = super(ShapedLayer, self).get_config()
        config['shape_in'] = self.get_shape_in()
        config['shape_out'] = self.get_shape_out()
        config['elementwise'] = self._elementwise
        return config

    def get_options(self):
        return ([], self.get_config())

class RecurrentLayer(Layer):

    def __init__(self, *args, **kwargs):
        self.stateful = kwargs.get('stateful', False)
        super(RecurrentLayer, self).__init__(*args, **kwargs)

        self.states = None

    def get_initial_states(self, X, shape_index=1):
        if self.stateful:
            N = X.batch_size
        else:
            N = T.shape(X)[shape_index]
        if self.stateful:
            if not N:
                raise Exception('Must set batch size for input')
            else:
                return [T.zeros((N, self.get_shape_out()))]
        return [T.alloc(0, (N, self.get_shape_out()), unbroadcast=shape_index)]

    def reset_states(self):
        if self.states is not None:
            for state in self.states:
                T.set_value(state, T.get_value(state) * 0)

    def reset_state(self, i):
        if self.states is not None:
            T.set_value(self.states[i], T.get_value(self.states[i]) * 0)


    def recurrent_forward(self, *inputs, **kwargs):
        raw_inputs = [x.get_placeholder() for x in inputs]
        raw_output = self._forward(*raw_inputs)
        return [Data.from_placeholder(
            raw_output,
            self.get_shape_out(),
            inputs[0].batch_size,
            sequence=True
        )]

    def step(self, X, state):
        out, state = self._step(X.get_placeholder(), state, [])
        return Data.from_placeholder(out, self.get_shape_out(), X.batch_size), state

    def _forward(self, X):
        S, N, D = T.shape(X)

        if self.stateful:
            if self.states is None:
                self.states = self.get_initial_states(N)
            hidden, state = self.states
        else:
            hidden, state = self.get_initial_states(X)

        _, output, new_state = T.rnn(self._step,
                              [X],
                              [hidden, state])
        if self.stateful:
            for state, ns in zip(self.states, new_state):
                self.updates.append((state, ns))
        return output

    def is_recurrent(self):
        return True

    def tie(self, node):
        new_node = self.copy(keep_params=True)
        for key, val in node.parameters.items():
            new_node.set_parameter_value(key, val)
        return new_node
