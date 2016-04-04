from .node import ShapedNode
from .binary import BinaryOpNode
from .exceptions import ShapeException
from .data import Data

class Chain(BinaryOpNode):

    def __init__(self, left, right):
        super(Chain, self).__init__(left, right)
        self.infer_shape()

    def get_network_inputs(self):
        inputs = []
        dups = set()
        for input in (self.left.get_network_inputs() + self.right.get_network_inputs()):
            if input not in dups:
                inputs.append(input)
                dups.add(input)
        return inputs

    def is_input(self):
        return self.left.is_input()

    def get_inputs(self):
        return self.left.get_inputs()

    def get_outputs(self, **kwargs):
        return self.forward(*self.get_inputs(), **kwargs)

    def forward(self, *left_input, **kwargs):
        right_input = self.left.forward(*left_input, **kwargs)
        right_input = right_input + self.right.get_inputs()
        right_output = self.right.forward(*right_input, **kwargs)
        return right_output

    def step(self, X, state):
        raise NotImplementedError

    def set_shape_in(self, shape_in):
        self.left.set_shape_in(shape_in)
        self.infer_shape()

    def set_shape_out(self, shape_out):
        self.right.set_shape_out(shape_out)

    def get_shape_in(self):
        return self.left.get_shape_in()

    def get_shape_out(self):
        return self.right.get_shape_out()

    def get_batch_size(self):
        return self.left.get_batch_size()

    def infer_shape(self):
        self.left.infer_shape()

        self.set_batch_size(self.left.get_batch_size())

        left_out = self.left.get_shape_out()
        right_in = self.right.get_shape_in()
        if left_out is not None and right_in is not None:
            if left_out != right_in:
                raise ShapeException(self.right, left_out)
        self.right.set_shape_in(left_out)
        self.right.infer_shape()

    def __str__(self):
        return "%s >> %s" % (self.left, self.right)

class Concatenate(BinaryOpNode):

    def __init__(self, left, right):
        super(Concatenate, self).__init__(left, right)
        self.infer_shape()

    def get_network_inputs(self):
        inputs = []
        dups = set()
        for input in (self.left.get_network_inputs() + self.right.get_network_inputs()):
            if input not in dups:
                inputs.append(input)
                dups.add(input)
        return inputs

    def is_input(self):
        return self.left.is_input() and self.right.is_input()

    def get_inputs(self):
        return (self.left.get_inputs(), self.right.get_inputs())

    def get_output(self, **kwargs):
        return self.forward(*self.get_inputs(), **kwargs)

    def forward(self, left_input, right_input, **kwargs):
        left_output, right_output = self.left.forward(*left_input), self.right.forward(*right_input)
        return [Data.concatenate(left_output + right_output)]

    def step(self, X, state):
        left_state, right_state = state
        left, left_state = self.left.step(X, left_state)
        right, right_state = self.right.step(left, right_state)
        return right, (left_state, right_state)

    def set_shape_in(self, shape_in):
        self.left.set_shape_in(shape_in)
        self.infer_shape()

    def set_shape_out(self, shape_out):
        self.right.set_shape_out(shape_out)

    def get_shape_in(self):
        return self.left.get_shape_in()

    def get_shape_out(self):
        return self.left.get_shape_out() + self.right.get_shape_out()

    def set_batch_size(self, batch_size):
        self.left.set_batch_size(batch_size)
        self.right.set_batch_size(batch_size)

    def get_batch_size(self):
        return self.left.get_batch_size()

    def infer_shape(self):
        pass

    def __str__(self):
        return "(%s) | (%s)" % (self.left, self.right)

class Index(ShapedNode):

    def __init__(self, index):
        super(Index, self).__init__()
        self.index = index

    def forward(self, X, **kwargs):
        return [Data.index(X, self.index)]

    # Shape inference

    def _infer(self, shape_in):
        return shape_in
