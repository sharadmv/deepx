from .. import backend as T
from .node import ShapedNode

class RecurrentNode(ShapedNode):

    def __init__(self, *args, **kwargs):
        super(RecurrentNode, self).__init__(*args, **kwargs)

        self.stateful = kwargs.get('stateful', False)
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

    def reset_state(self, i):
        if self.states is not None:
            T.set_value(self.states[i], T.get_value(self.states[i]) * 0)

    def _recurrent_forward(self, X, **kwargs):
        return self._forward(X)

    def is_recurrent(self):
        return True
