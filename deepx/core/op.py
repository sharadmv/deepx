import six
from abc import ABCMeta, abstractmethod

from deepx.backend import T

@six.add_metaclass(ABCMeta)
class Op(object):

    def __init__(self):
        self.parameters = {}
        self.device = T.get_current_device()
        self.shape_in, self.shape_out = None, None

    def get_parameters(self):
        return list(self.parameters.values())

    def get_parameter(self, key):
        return self.parameters[key]

    def set_parameter(self, key, value):
        self.parameters[key] = value

    def create_parameter(self, name, dims, initial_value=None):
        if initial_value is None:
            value = T.random_normal(dims)
        else:
            value = initial_value
        self.set_parameter(name, T.variable(value))

    def __call__(self, *inputs):
        if self.get_shape_in() is None:
            in_shape = [T.get_shape(input) for input in inputs]
            self.set_shape_in(in_shape)
            self.shape_inference()
        output = self.forward(*inputs)
        if isinstance(output, list) or isinstance(output, tuple):
            if len(output) == 1:
                return output[0]
        return output

    def get_shape_out(self):
        return self.shape_out

    def get_shape_in(self):
        return self.shape_in

    def set_shape_in(self, shape_in):
        self.shape_in = shape_in

    def set_shape_out(self, shape_out):
        self.shape_out = shape_out

    def compose(self, op):
        from deepx.core.compose import Compose
        return Compose(self, op)

    def add(self, op):
        from deepx.core.arithmetic import Add
        return Add(self, op)

    def sub(self, op):
        from deepx.core.arithmetic import Sub
        return Sub(self, op)

    def mul(self, op):
        from deepx.core.arithmetic import Mul
        return Mul(self, op)

    def div(self, op):
        from deepx.core.arithmetic import Div
        return Div(self, op)

    def __add__(self, op):
        return self.add(op)

    def __radd__(self, op):
        return self.add(op)

    def __sub__(self, op):
        return self.sub(op)

    def __rsub__(self, op):
        return op.sub(self)

    def __mul__(self, op):
        return self.mul(op)

    def __rmul__(self, op):
        return self.mul(op)

    def __div__(self, op):
        return self.div(op)

    def __rdiv__(self, op):
        return op.div(self)

    def __rshift__(self, op):
        return self.compose(op)

    def __rrshift__(self, op):
        return self.compose(op)

    def __repr__(self):
        return "{op_name}({shape_in}, {shape_out})".format(
            op_name=self.__class__.__name__,
            shape_in=self.get_shape_in(),
            shape_out=self.get_shape_out()
        )

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def shape_inference(self):
        pass

    @abstractmethod
    def is_initialized(self):
        pass
