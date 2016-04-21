from .node import Node

class BinaryOpNode(Node):

    def __init__(self, left, right):
        super(BinaryOpNode, self).__init__()
        self.left = left
        self.right = right
        self.infer_shape()

    def can_initialize(self):
        return self.left.can_initialize() and self.right.can_initialize()

    def reinitialize(self, **kwargs):
        self.left.reinitialize(**kwargs)
        self.right.reinitialize(**kwargs)

    def initialize(self, **kwargs):
        self.left.initialize(**kwargs)
        self.right.initialize(**kwargs)

    def get_updates(self):
        return self.left.get_updates() + self.right.get_updates()

    def get_inputs(self):
        raise NotImplementedError

    def get_graph_inputs(self):
        inputs = []
        dups = set()
        for input in (self.left.get_graph_inputs() + self.right.get_graph_inputs()):
            if input not in dups:
                inputs.append(input)
                dups.add(input)
        return inputs

    def get_graph_parameters(self):
        if self.frozen:
            return []
        parameters = []
        dups = set()
        for parameter in (self.left.get_graph_parameters() + self.right.get_graph_parameters()):
            if parameter not in dups:
                parameters.append(parameter)
                dups.add(parameter)
        return parameters

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

    def set_batch_size(self, batch_size):
        self.left.set_batch_size(batch_size)
        self.right.set_batch_size(batch_size)

    def is_configured(self):
        return self.left.is_configured() and self.right.is_configured()

    def get_parameters(self):
        return (self.left.get_parameters(), self.right.get_parameters())

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

    def reset_states(self):
        self.left.reset_states()
        self.right.reset_states()

    def reset_state(self, i):
        self.left.reset_state(i)
        self.right.reset_state(i)

    def tie(self, node):
        new_node = self.__class__(self.left.tie(node.left), self.right.tie(node.right))
        new_node.infer_shape()
        return new_node
