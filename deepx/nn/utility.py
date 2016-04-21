from ..core import Layer

def shape_identity(shape_in):
    return shape_in

class Lambda(Layer):

    def __init__(self, func, shape_func=shape_identity):
        super(Lambda, self).__init__()
        self.func = func
        self.shape_func = shape_func

    def initialize(self):
        pass

    def _infer(self, shape_in):
        return self.shape_func(shape_in)

    def _forward(self, X, **kwargs):
        return self.func(X)
