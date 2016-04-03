from .node import Node
from .exceptions import ShapeException

class BinaryOpNode(Node):

    def __init__(self, left, right):
        super(BinaryOpNode, self).__init__()
        self.left = left
        self.right = right

    def get_updates(self):
        return self.left.get_updates() + self.right.get_updates()

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

    def get_initial_states(self, *args, **kwargs):
        return self.left.get_initial_states(*args, **kwargs), self.right.get_initial_states(*args, **kwargs)

    def set_parameter_tree(self, params):
        left_params, right_params = params
        self.left.set_parameter_tree(left_params)
        self.right.set_parameter_tree(right_params)

    def copy(self, keep_params=False, **kwargs):
        old_params = self.get_parameter_tree()
        node = self.__class__(self.left.copy(**kwargs), self.right.copy(**kwargs))
        if keep_params:
            node.set_parameter_tree(old_params)
        node.infer_shape()
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
        return self.left.get_inputs()

    def get_outputs(self, **kwargs):
        return self.forward(*self.get_inputs(), **kwargs)

    def forward(self, *left_input, **kwargs):
        right_input = self.left.forward(*left_input, **kwargs)
        right_input = right_input + self.right.get_inputs()
        right_output = self.right.forward(*right_input, **kwargs)
        return right_output

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
        return self.right.get_shape_out()

    def set_batch_size(self, batch_size):
        self.left.set_batch_size(batch_size)
        self.right.set_batch_size(batch_size)

    def get_batch_size(self):
        return self.left.get_batch_size()

    def reset_states(self):
        self.left.reset_states()
        self.right.reset_states()

    def reset_state(self, i):
        self.left.reset_states(i)
        self.right.reset_states(i)

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
