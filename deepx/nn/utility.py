from ..node import Node

def shape_identity(shape_in):
    return shape_in

class Lambda(Node):

    def __init__(self, func, shape_func=shape_identity):
        super(Lambda, self).__init__(func, shape_func=shape_func)
        self.func = func
        self.shape_func = shape_func

    def can_initialize(self):
        return True

    def _infer(self, shape_in):
        return self.shape_func(shape_in)

    def _forward(self, X):
        return self.func(X)
