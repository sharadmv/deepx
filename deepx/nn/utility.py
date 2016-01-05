from ..node import Node

def shape_identity(shape_in):
    return shape_in

class Lambda(Node):

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.shape_func = kwargs.pop('shape_func', shape_identity)

        super(Lambda, self).__init__(*args, **kwargs)

    def _infer(self, shape_in):
        return self.shape_func(shape_in)

    def _forward(self, X):
        return self.func(X)

    def copy(self):
        return Lambda(self.func, shape_func=self.shape_func)
