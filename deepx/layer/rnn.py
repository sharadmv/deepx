import theano.tensor as T

from layer import Layer
from model import Data

class RecurrentLayer(Layer):

    def get_in_var(self):
        return T.matrix()

    def get_layer_var(self):
        return T.matrix()

    def is_recurrent(self):
        return True

    def forward(self, X, previous):
        return [Data(self._forward(X.X, previous.X))]

class StatefulRecurrentLayer(RecurrentLayer):

    def get_layer_var(self, ndim):
        if ndim == 1:
            return (T.matrix(), T.matrix())
        if ndim == 2:
            return (T.tensor3(), T.tensor3())
        if ndim == 3:
            return (T.tensor4(), T.tensor4())
        raise Exception("Data too dimensional")

    def is_recurrent(self):
        return True

    def forward(self, X, previous):
        return [Data(self._forward(X.X, previous))]
