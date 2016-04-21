from .. import backend as T
from ..core import Layer, Data

class Loss(Layer):

    def __init__(self):
        self.y = None
        super(Loss, self).__init__()

    def _infer(self, shape_in):
        return shape_in.copy(dim=())

    def initialize(self):
        pass

    def get_inputs(self):
        if self.y is None:
            self.y = Data(self.get_shape_in()[0], name='y')
        return [self.y]

    def _sequence_loss(self, X):
        return T.mean(X)

    def forward(self, inputs, **kwargs):
        X, y = inputs
        if X.is_sequence():
            def step(*inputs):
                return self._loss(*inputs)
            return [Data(self.get_shape_out()[0],
                        placeholder=T.scan(step, [X.get_placeholder(), y.get_placeholder()])[0])]
        return [Data(self.get_shape_out()[0],
                    placeholder=self._forward(X.get_placeholder(), y.get_placeholder()))]

    def _forward(self, X, y, **kwargs):
        return self._loss(X, y)

    def __str__(self):
        return self.__class__.__name__
