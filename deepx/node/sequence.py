import numpy as np
from .. import backend as T
from .node import Node
from ..util import pack_tuple, unpack_tuple

def Sequence(data, max_length=None):
    batch_size = data.batch_size
    var = T.make_sequence(data.get_data(), max_length)
    data = data.next(var, data.get_shape_out())
    data.sequence = True
    data.sequence_length = max_length
    data.batch_size = batch_size
    return data

class Generate(Node):

    def __init__(self, node, length=None):
        super(Generate, self).__init__()
        self.node = node
        self.length = length
        assert self.node.get_shape_out() == self.node.get_shape_in()

    def forward(self, X, **kwargs):
        states = self.node.get_initial_states(X.get_data(), shape_index=0)
        states, shape = unpack_tuple(states)

        def step(input, softmax, *states):
            state = pack_tuple(states, shape)
            out, state = self.node.step(X.next(input, self.get_shape_in()), state)
            out_softmax = out.get_data()
            out_softmax = T.log(out_softmax)
            out_softmax /= 0.1
            out_softmax = T.exp(out_softmax)
            out_sample = T.sample(out_softmax)
            states, _ = unpack_tuple(state)
            return [out_sample, out_softmax] + list(states)

        output, self.updates = T.generate(step, [X.get_data(), X.get_data()] + list(states), self.length)
        out = X.next(output[1], self.get_shape_out())
        out.sequence = True
        out.sequence_length = self.length
        return out

    def infer_shape(self):
        self.node.infer_shape()

    def get_input(self):
        return self.node.get_input()

    def get_shape_in(self):
        return self.node.get_shape_in()

    def get_shape_out(self):
        return self.node.get_shape_out()

    def get_state(self):
        return self.node.get_state()

    def set_state(self, state):
        self.node.set_state(state)

    def get_parameters(self):
        return self.node.get_parameters()

    def __str__(self):
        return "Generate(%s)" % str(self.node)

class RecurrentNode(Node):

    def __init__(self, stateful=False):
        super(RecurrentNode, self).__init__()

        self.stateful = stateful
        self.states = None

    def get_initial_states(self, X, shape_index=1):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.stateful:
            N = self.get_batch_size()
        else:
            N = T.shape(X)[shape_index]
        if self.stateful:
            if not N:
                raise Exception('Must set batch size for input')
            else:
                return [T.zeros((N, self.get_shape_out())),
                        T.zeros((N, self.get_shape_out()))]
        return [T.alloc(0, (N, self.get_shape_out()), unbroadcast=1),
                T.alloc(0, (N, self.get_shape_out()), unbroadcast=1)]

    def reset_states(self):
        for state in self.states:
            T.set_value(state, T.get_value(state) * 0)


    def forward(self, X, **kwargs):
        return super(RecurrentNode, self).forward(X, **kwargs)

    def recurrent_forward(self, X):
        output = self._forward(X.get_data())
        return X.next(output, self.get_shape_out())

    def is_recurrent(self):
        return True
