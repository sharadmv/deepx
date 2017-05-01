from .node import Node

__all__ = ['Chain']

class Chain(Node):

    def __init__(self, left, right):
        super(Chain, self).__init__()
        self.left, self.right = left, right
        self.infer_shape(self.get_dim_in())
        if self.is_initialized():
            self.initialize()

    def forward(self, *args):
        return self.right(self.left(*args))

    def infer_shape(self, shape):
        self.left.infer_shape(shape)
        self.right.infer_shape(self.left.get_dim_out())

    def get_dim_in(self):
        return self.left.get_dim_in()

    def get_dim_out(self):
        return self.right.get_dim_out()

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
