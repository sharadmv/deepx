from .node import Node

class BinaryOpNode(Node):

    def __init__(self, left, right):
        super(BinaryOpNode, self).__init__()
        self.left = left
        self.right = right
        self.infer_shape()

    def get_graph_parameters(self):
        return self.left.get_graph_parameters() + self.right.get_graph_parameters()

    def get_graph_inputs(self):
        return self.left.get_graph_inputs() + self.right.get_graph_inputs()

    def get_graph_updates(self, **kwargs):
        return self.left.get_graph_updates(**kwargs) + self.right.get_graph_updates(**kwargs)

    def reset_states(self):
        self.left.reset_states()
        self.right.reset_states()

    def reset_state(self, i):
        self.left.reset_state(i)
        self.right.reset_state(i)

    def initialize(self, **kwargs):
        self.left.initialize(**kwargs)
        self.right.initialize(**kwargs)

    def reinitialize(self, **kwargs):
        self.left.reinitialize(**kwargs)
        self.right.reinitialize(**kwargs)
