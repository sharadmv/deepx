from abc import abstractmethod
from .unary import UnaryNode
from .node import Node

__all__ = ["FunctionalNode", "HOF"]

class FunctionalNode(UnaryNode):

    @abstractmethod
    def func(self, inputs):
        pass

    def get_outputs(self, *args, **kwargs):
        net = self.func(args)
        return net.get_outputs(**kwargs)

    @abstractmethod
    def get_shapes_out(self):
        pass

    @abstractmethod
    def get_num_outputs(self):
        pass

class HigherOrderNode(Node):

    def __init__(self, hof, *args, **kwargs):
        super(HigherOrderNode, self).__init__(*args, **kwargs)
        self.hof = hof

    def get_graph_inputs(self):
        return []

    def get_shapes_out(self):
        return self.node.get_shapes_out()

    def func(self, inputs):
        return self.hof(inputs)

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1


HOF = HigherOrderNode
