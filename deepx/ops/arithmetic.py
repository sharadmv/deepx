from abc import abstractmethod
import six

from .. import T
from ..core import ShapedNode, Shape, Data

__all__ = ['Concatenate', 'Max', 'Sum', 'Pow', 'Add', 'Mul', 'Prod', 'Sub', 'Div']

class SimpleOperator(ShapedNode):

    def __init__(self):
        super(SimpleOperator, self).__init__(None, None)

    def inputs(self):
        return []


class Concatenate(SimpleOperator):

    def get_outputs(self, *inputs, **kwargs):
        return [Data.concatenate(inputs)]

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        if self.shapes_in is not None:
            self.shapes_out = [Shape.concatenate(self.shapes_in)]

    def __repr__(self):
        return "Concatenate()"

    def __str__(self):
        return repr(self)


class Sum(SimpleOperator):

    def get_outputs(self, input, **kwargs):
        raw_input = input.get_placeholder()
        raw_output = T.sum(raw_input, axis=-1)
        return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        if self.shapes_in is not None:
            shape_in = self.shapes_in[0]
            self.shapes_out = [shape_in.copy(
                dim=shape_in.get_dim()[:-1]
            )]

    def __repr__(self):
        return "Sum()"

    def __str__(self):
        return repr(self)


class Max(SimpleOperator):

    def get_outputs(self, input, **kwargs):
        raw_input = input.get_placeholder()
        raw_output = T.max(raw_input, axis=-1)
        return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        if self.shapes_in is not None:
            shape_in = self.shapes_in[0]
            self.shapes_out = [shape_in.copy(
                dim=shape_in.get_dim()[:-1]
            )]

    def __repr__(self):
        return "Max()"

    def __str__(self):
        return repr(self)

class UnaryOperator(SimpleOperator):

    def forward(self, X):
        return [self.op(X)]

    @abstractmethod
    def op(self, X):
        pass

    def infer_shape(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            self.set_shapes_out([shapes_in[0].copy()])

    def __str__(self):
        return repr(self)

class Pow(UnaryOperator):

    def __init__(self, pow):
        super(Pow, self).__init__()
        self.pow = pow

    def op(self, X):
        return T.pow(X, self.pow)

    def __repr__(self):
        return "Pow(%u)" % self.pow

class Exp(UnaryOperator):

    def op(self, X):
        return T.exp(X)

    def __repr__(self):
        return "Exp()"

class ArithmeticOperator(SimpleOperator):

    def outputs(self, *inputs, **kwargs):
        raw_output = six.moves.reduce(self.op, inputs)
        return [raw_output]

    def infer_shape(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            shapes_in = sorted(shapes_in, key=lambda x: len(x.get_dim()))
            self.set_shapes_out([shapes_in[0].copy()])

    def __repr__(self):
        return "%s()" % self.op_name

    def __str__(self):
        return repr(self)

class Add(ArithmeticOperator):
    op_name = 'Add'

    @staticmethod
    def op(x, y):
        return x + y

class Mul(ArithmeticOperator):
    op_name = 'Mul'

    @staticmethod
    def op(x, y):
        return x * y

Prod = Mul


class Sub(ArithmeticOperator):
    op_name = 'Sub'

    @staticmethod
    def op(x, y):
        return x - y

class Div(ArithmeticOperator):
    op_name = 'Div'

    @staticmethod
    def op(x, y):
        return x / y

class Index(SimpleOperator):

    def __init__(self, index):
        super(Index, self).__init__()
        self.index = index

    def get_outputs(self, input, **kwargs):
        raw_input = input.get_placeholder()
        if T.ndim(raw_input) == 2:
            raw_output = raw_input[:, self.index]
        if T.ndim(raw_input) == 3:
            raw_output = raw_input[:, :, self.index]
        return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        if self.shapes_in is not None:
            shape_in = self.shapes_in[0]
            self.shapes_out = [shape_in.copy(
                dim=shape_in.get_dim()[:-1]
            )]

    def __repr__(self):
        return "Index()"

    def __str__(self):
        return repr(self)
