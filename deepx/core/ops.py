import six

from .. import backend as T

from .node import Node
from .shape import Shape
from .binary import BinaryOpNode
from .exceptions import ShapeInError
from .data import Data

__all__ = ['Concatenate', 'Add', 'Prod', 'Sub', 'Div']

class SimpleOperator(Node):

    def get_graph_parameters(self):
        return []

    def get_graph_inputs(self):
        return []

    def get_graph_updates(self, **kwargs):
        return []

    def reset_states(self):
        return

    def reset_state(self, i):
        return

    def initialize(self, **kwargs):
        return

    def reinitialize(self, **kwargs):
        return

    def get_shapes_in(self):
        return self.shapes_in

    def get_shapes_out(self):
        return self.shapes_out

    def set_shapes_in(self, shapes_in):
        self.shapes_in = shapes_in

    def set_shapes_out(self, shapes_out):
        self.shapes_out = shapes_out

    def get_num_inputs(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is None:
            return None
        return len(shapes_in)

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

class ArithmeticOperator(SimpleOperator):

    def get_outputs(self, *inputs, **kwargs):
        raw_inputs = [d.get_placeholder() for d in inputs]
        raw_output = reduce(self.op, raw_inputs)
        return [Data(self.get_shapes_out()[0],
                     placeholder=raw_output)]

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            shapes_in = sorted(shapes_in, key=lambda x: len(x.get_dim()))
            self.shapes_out = [shapes_in[0].copy()]

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

class Index(Node):

    def __init__(self, index):
        super(Index, self).__init__()
        self.index = index

    def forward(self, X, **kwargs):
        return [Data.index(X, self.index)]

    def has_parameters(self):
        return False

    # Shape inference

    def _infer(self, shape_in):
        return shape_in.copy(sequence=False, max_length=None)

    def __str__(self):
        return "Index(%u)" % self.index
