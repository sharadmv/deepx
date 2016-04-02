from .. import backend as T

from ..core import Layer, Data

class Loss(Layer):

    def __init__(self):
        super(Loss, self).__init__()

    def _infer(self, shape_in):
        return ()

    def initialize(self):
        self.y = Data(self.get_shape_in(), name='y')

    def get_inputs(self):
        return [self.y]

    def _forward(self, X, y):
        return self._loss(X, y)

    def __str__(self):
        return self.__class__.__name__
