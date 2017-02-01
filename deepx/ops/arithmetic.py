import six

from .. import T
from ..core import ShapedNode, Shape, Data

__all__ = ['Concatenate', 'Max', 'Sum', 'Pow', 'Add', 'Prod', 'Sub', 'Div']

class SimpleOperator(ShapedNode):

    def __init__(self):
        super(SimpleOperator, self).__init__(None, None)

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

class Pow(SimpleOperator):

    def __init__(self, pow):
        super(Pow, self).__init__()
        self.pow = pow

    def get_outputs(self, input, **kwargs):
        raw_input = input.get_placeholder()
        raw_output = T.pow(raw_input, self.pow)
        return [Data(self.get_shapes_out()[0], placeholder=raw_output)]

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        if self.shapes_in is not None:
            shape_in = self.shapes_in[0]
            self.shapes_out = [shape_in.copy()]

    def __repr__(self):
        return "Pow(%u)" % self.pow

    def __str__(self):
        return repr(self)

class ArithmeticOperator(SimpleOperator):

    def forward(self, *inputs, **kwargs):
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

class Prod(ArithmeticOperator):
    op_name = 'Prod'

    @staticmethod
    def op(x, y):
        return x * y


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
        return "Max()"

    def __str__(self):
        return repr(self)
