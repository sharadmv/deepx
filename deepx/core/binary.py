from .node import Node
from .exceptions import ShapeException

class BinaryOpNode(Node):

    def __init__(self, left, right):
        super(BinaryOpNode, self).__init__()
        self.left = left
        self.right = right

    def get_inputs(self):
        return self.left.get_inputs(), self.right.get_inputs()

    def get_outputs(self, **kwargs):
        left_input, right_input = self.left.get_inputs(), self.right.get_inputs()
        return self.forward(left_input, right_input, **kwargs)

    def has_parameters(self):
        return self.left.has_parameters() or self.right.has_parameters()

    def set_shape_in(self, shape_in):
        raise NotImplementedError

    def set_shape_out(self, shape_out):
        raise NotImplementedError

    def get_shape_in(self):
        raise NotImplementedError

    def get_shape_out(self):
        raise NotImplementedError

    def is_configured(self):
        return self.left.is_configured() and self.right.is_configured()

    def get_parameters(self):
        if self.frozen:
            return []
        params = []
        dups = set()
        if self.left.has_parameters():
            for param in self.left.get_parameters():
                dups.add(param)
                params.append(param)
        if self.right.has_parameters():
            for param in self.right.get_parameters():
                if param not in dups:
                    params.append(param)
        return params

    def get_parameter_tree(self):
        return (self.left.get_parameter_tree(), self.right.get_parameter_tree())

    def get_state(self, **kwargs):
        return (self.left.get_state(**kwargs), self.right.get_state(**kwargs))

    def set_state(self, state):
        left_state, right_state = state
        self.left.set_state(left_state)
        self.right.set_state(right_state)

    def set_parameter_tree(self, params):
        left_params, right_params = params
        self.left.set_parameter_tree(left_params)
        self.right.set_parameter_tree(right_params)

    def copy(self, keep_params=False, **kwargs):
        old_params = self.get_parameter_tree()
        node = self.__class__(self.left.copy(**kwargs), self.right.copy(**kwargs))
        if keep_params:
            node.infer_shape()
            node.set_parameter_tree(old_params)
        return node

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
        return self.left.get_inputs() + self.right.get_inputs()

    def get_outputs(self, **kwargs):
        return self.forward(*self.get_inputs(), **kwargs)

    def forward(self, *left_inputs, **kwargs):
        left_outputs = self.left.forward(*left_inputs, **kwargs)
        right_outputs = self.right.forward(*left_outputs, **kwargs)
        return right_outputs

    def set_shape_in(self, shape_in):
        self.left.set_shape_in(shape_in)
        self.infer_shape()

    def set_shape_out(self, shape_out):
        self.right.set_shape_out(shape_out)

    def get_shape_in(self):
        return self.left.get_shape_in()

    def get_shape_out(self):
        return self.right.get_shape_out()

    def infer_shape(self):
        self.left.infer_shape()
        left_out = self.left.get_shape_out()
        right_in = self.right.get_shape_in()
        if right_in is not None:
            if left_out != right_in:
                raise ShapeException(self.right, left_out)
        self.right.set_shape_in(left_out)
        self.right.infer_shape()

    def __str__(self):
        return "%s >> %s" % (self.left, self.right)
