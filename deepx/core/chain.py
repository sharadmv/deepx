from .node import Node

__all__ = ['Chain']

class Chain(Node):

    def __init__(self, left, right):
        super(Chain, self).__init__()
        self.left, self.right = left, right

    def forward(self, *args):
        return self.right(self.left(*args))

    def infer_shape(self, *args):
        pass

    def is_initialized(self):
        return self.left.is_initialized() and self.right.is_initialized()

    def initialize(self):
        self.left.initialize()
        self.right.initialize()

    def get_parameters(self):
        return self.left.get_parameters() + self.right.get_parameters()

    def __repr__(self):
        return "%s >> %s" % (self.left, self.right)

    def __str__(self):
        return repr(self)
