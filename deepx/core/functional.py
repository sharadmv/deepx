from abc import abstractmethod
from .unary import UnaryNode
from .node import ShapedNode

__all__ = ["FunctionalNode", "HOF"]

def HOF(func):
    def foo(node):
        return HigherOrderNode(node, func)
    return foo

class HigherOrderNode(ShapedNode):

    def __init__(self, node, func, **kwargs):
        super(HigherOrderNode, self).__init__(None, None, **kwargs)
        self.node = node
        self.func = func

    def infer_shape(self):
        self.node.set_shapes_in(self.get_shapes_in())
        self.node.infer_shape()
        self.set_shapes_out(self.node.get_shapes_out())

    def inputs(self):
        return self.node.inputs()

    def outputs(self, *inputs):
        new_node = self.func(inputs, self.node)

    def overrides_chain(self):
        return True

    def rchain(self, node):
        return self.func(node, self.node)

    def forward(self, X):
        return X

    def __repr__(self):
        return "HOF(%s)" % self.node

# HOF = HigherOrderNode
