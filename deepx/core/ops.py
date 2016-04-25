from .node import Node
from .shape import Shape
from .binary import BinaryOpNode
from .exceptions import ShapeInError
from .data import Data

__all__ = ["Chain", "Concatenate"]

class Chain(BinaryOpNode):

    def __init__(self, left, right):
        super(Chain, self).__init__(left, right)
        if left.get_num_outputs() != right.get_num_inputs():
            raise TypeError("Cannot chain %s and %s" % (left, right))

    def get_outputs(self, *inputs, **kwargs):
        left_out = self.left.get_outputs(*inputs, **kwargs)
        right_out = self.right.get_outputs(*left_out, **kwargs)
        return right_out

    def get_shapes_in(self):
        return self.left.get_shapes_in()

    def get_shapes_out(self):
        return self.right.get_shapes_out()

    def set_shapes_in(self, shapes_in):
        self.left.set_shapes_in(shapes_in)

    def set_shapes_out(self, shapes_out):
        self.right.set_shapes_out(shapes_in)

    def get_num_inputs(self):
        return self.left.get_num_inputs()

    def get_num_outputs(self):
        return self.right.get_num_outputs()

    def infer_shape(self):
        self.left.infer_shape()
        self.right.infer_shape()
        if self.left.get_shapes_out() is not None or self.left.get_shapes_out() == self.right.get_shapes_out():
            self.right.set_shapes_in(self.left.get_shapes_out())
            self.right.infer_shape()

    def __repr__(self):
        return "%s >> %s" % (self.left, self.right)

    def __str__(self):
        return repr(self)

class Concatenate(Node):

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

    def get_outputs(self, *inputs, **kwargs):
        return [Data.concatenate(inputs)]

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

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        if self.shapes_in is not None:
            self.shapes_out = [Shape.concatenate(self.shapes_in)]

    def __repr__(self):
        return "Concatenate()"

    def __str__(self):
        return repr(self)

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
