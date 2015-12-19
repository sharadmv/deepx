import theano
import numpy as np
import theano.tensor as T
from node import Node, Data

class RecurrentNode(Node):

    def recurrent_forward(self, X, previous):
        out = self.forward(X, previous)
        return out, out

    def forward(self, X, previous):
        return Data(self._forward(X.get_data(), previous), self.shape_out)

    def is_recurrent(self):
        return True

    def get_previous_zeros(self, N):
        return T.alloc(np.array(0).astype(theano.config.floatX), N, self.get_shape_out())
