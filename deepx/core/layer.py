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
            outputs.append(Data.from_placeholder(raw_output, dim, batch_size))
        return outputs

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

class ShapedLayer(Layer):

    def __init__(self, shape_in=None, shape_out=None, **kwargs):

        self._elementwise = False
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
        return config

    def get_options(self):
        return ([], self.get_config())
