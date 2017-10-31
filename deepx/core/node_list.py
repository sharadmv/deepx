from .node import Node

class NodeTuple(Node):

    def __init__(self, nodes):
        self.nodes = nodes

    def forward(self, *args):
        return tuple(node.forward(*args) for node in self.nodes)

    def infer_shape(self, *shapes_in):
        for node in self.nodes:
            node.infer_shape(*shapes_in)

    def is_initialized(self):
        return all(node.is_initialized() for node in self.nodes)

    def initialize(self):
        for node in self.nodes:
            node.initialize()

    def get_parameters(self):
        return [param for node in self.nodes for param in node.get_parameters()]

    def get_dim_in(self):
        dim_in = self.nodes[0].get_dim_in()
        for node in self.nodes:
            node_in = node.get_dim_in()
            if node_in is not None and node_in != dim_in:
                raise Exception()
        return dim_in

    def get_dim_out(self):
        return tuple(node.get_dim_out() for node in self.nodes)

    def __repr__(self):
        return "(%s)" % ", ".join(map(str, self.nodes))
