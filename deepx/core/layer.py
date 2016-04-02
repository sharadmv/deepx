from .. import backend as T
from .node import ShapedNode
from .data import Data
from .initialization import initialize_weights

class Layer(ShapedNode):

    def __init__(self, *args, **kwargs):
        self.weight_init = kwargs.pop('weight_init', 'default')
        self.parameters = {}
        self.initialized = False

        super(Layer, self).__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        raw_inputs = [data.get_placeholder() for data in inputs]
        batch_size = inputs[0].batch_size

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

    def init_parameter(self, name, shape):
        parameter = T.variable(
            initialize_weights(shape, self.weight_init),
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

    def has_parameters(self):
        return True

    def get_parameters(self):
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

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, str(self.get_shape_in()), str(self.get_shape_out()))

class ShapedLayer(Layer):

    def __init__(self, *args):
        assert len(args) <= 2

        self._elementwise = False
        if len(args) == 2:
            shape_in, shape_out = args
        elif len(args) == 1:
            shape_in, shape_out = None, args[0]
        else:
            shape_in, shape_out = None, None
            self._elementwise = True

        super(ShapedLayer, self).__init__(shape_in=shape_in, shape_out=shape_out)

    def is_elementwise(self):
        return self._elementwise
