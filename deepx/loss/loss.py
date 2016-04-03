from .. import backend as T
from ..core import Layer, Data

class Loss(Layer):

    def __init__(self):
        self.y = None
        super(Loss, self).__init__()

    def _infer(self, shape_in):
        return ()

    def initialize(self):
        pass

    def get_inputs(self):
        if self.y is None:
            self.y = Data(self.get_shape_in(), name='y')
        return [self.y]

    def forward(self, X, y, **kwargs):
        if X.is_sequence():
            y = self.y = Data(self.get_shape_in(), name='y', sequence=True)
        output = super(Loss, self).forward(X, y, **kwargs)[0]
        if output.is_sequence():
            output = [Data.from_placeholder(
                self._sequence_loss(output.get_placeholder()),
                self.get_shape_out(),
                None,
                sequence=False
            )]
        return output

    def _sequence_loss(self, X):
        return T.mean(X)

    def _forward(self, X, y):
        return self._loss(X, y)

    def __str__(self):
        return self.__class__.__name__
