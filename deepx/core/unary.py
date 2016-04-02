from .node import Node

class UnaryOpNode(Node):

    def __init__(self, node):
        super(UnaryOpNode, self).__init()
        self.node = node

    def get_inputs(self):
        return self.node.get_inputs()

    def get_outputs(self):
        raise NotImplementedError

    def is_input(self):
        return self.node.is_input()

    def get_parameters(self):
        return self.node.get_parameters()
