import numpy as np
from .. import backend as T
from .node import Node
from ..util import pack_tuple, unpack_tuple

class RecurrentNode(Node):

    def __init__(self, stateful=False):
        super(RecurrentNode, self).__init__()

        self.stateful = stateful
        self.states = None

    def get_initial_states(self, X, shape_index=1):
        if self.stateful:
            N = self.get_batch_size()
        else:
            N = T.shape(X)[shape_index]
        if self.stateful:
            if not N:
                raise Exception('Must set batch size for input')
            else:
                return [T.zeros((N, self.get_shape_out()))]
        return [T.alloc(0, (N, self.get_shape_out()), unbroadcast=shape_index)]

    def reset_states(self):
        if self.states is not None:
            for state in self.states:
                T.set_value(state, T.get_value(state) * 0)

    def recurrent_forward(self, X, **kwargs):
        return X.next(self._forward(X.get_data()), self.get_shape_out())

    def is_recurrent(self):
        return True
