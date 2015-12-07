import numpy as np
import theano
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

    def get_sequence_var(self):
        return T.tensor3()

    def forward(self, X, previous):
        return [Data(self._forward(X.get_data(), previous.get_data()))]

class StatefulRecurrentLayer(RecurrentLayer):

    def alloc(self, N):
        return (T.alloc(np.array(0).astype(theano.config.floatX), N, self.n_out),
                T.alloc(np.array(0).astype(theano.config.floatX), N, self.n_out))

    def get_layer_var(self):
        return (T.matrix('output'), T.matrix('state'))

    def is_recurrent(self):
        return True

    def forward(self, X, previous):
        return [Data(self._forward(X.get_data(), previous))]
