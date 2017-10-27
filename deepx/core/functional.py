from abc import abstractmethod
from .node import Node

__all__ = ["FunctionalNode", "HOF"]

def HOF(func):
    def foo(node):
        return HigherOrderNode(node, func)
    return foo

class HigherOrderNode(Node):

    def __init__(self, node, func, **kwargs):
        super(HigherOrderNode, self).__init__(None, None, **kwargs)
        self.node = node
        self.func = func

    def infer_shape(self):
        self.node.set_shapes_in(self.get_shapes_in())
        self.node.infer_shape()
        self.set_shapes_out(self.node.get_shapes_out())

    def forward(self, X):
        return X

    def __repr__(self):
        return "HOF(%s)" % self.node


    def is_initialized(self):
        return self.node.is_initialized()

    def initialize(self):
        self.node.initialize()

    def get_dim_in(self):
        return self.node.get_dim_in()
        pass

    def get_dim_out(self):
        return self.node.get_dim_out()

# HOF = HigherOrderNode
