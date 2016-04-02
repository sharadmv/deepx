from ..core import Layer

def shape_identity(shape_in):
    return shape_in

class Lambda(Layer):

    def __init__(self, func, shape_func=shape_identity):
        super(Lambda, self).__init__(func=func, shape_func=shape_func)
        self.func = func
        self.shape_func = shape_func

    def initialize(self):
        pass

    def _infer(self, shape_in):
        return self.shape_func(shape_in)

    def _forward(self, X):
        return self.func(X)

    def copy(self, **kwargs):
        return Lambda(self.func, self.shape_func)
