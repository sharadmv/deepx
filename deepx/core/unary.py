from abc import abstractmethod
from .node import Node

__all__ = ["UnaryNode"]

class UnaryNode(Node):

    def __init__(self, node):
        super(UnaryNode, self).__init__()
        self.node = node

    @abstractmethod
    def get_outputs(self, *args, **kwargs):
        pass

    def get_graph_inputs(self):
        return self.node.get_graph_inputs()

    def get_graph_parameters(self):
        return self.node.get_graph_parameters()

    def get_graph_updates(self, **kwargs):
        return self.node.get_graph_updates(**kwargs)

    def reset_states(self):
        self.node.reset_states()

    def reset_state(self, i):
        self.node.reset_state(i)

    def initialize(self, **kwargs):
        self.node.initialize(**kwargs)

    def reinitialize(self, **kwargs):
        self.node.reinitialize(**kwargs)

    # Shape inference

    def set_shapes_in(self, shapes_in):
        self.node.set_shapes_in(shapes_in)

    def set_shapes_out(self, shapes_out):
        self.node.set_shapes_out(shapes_out)

    def get_shapes_in(self):
        return self.node.get_shapes_in()

    @abstractmethod
    def get_shapes_out(self):
        pass

    def get_num_inputs(self):
        return self.node.get_num_inputs()

    @abstractmethod
    def get_num_outputs(self):
        pass

    def infer_shape(self):
        self.node.infer_shape()

    def get_state(self, **kwargs):
        return self.node.get_state(**kwargs)

    def set_state(self, state):
        self.node.set_state(state)
