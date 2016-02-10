import numpy as np
from .. import backend as T
from .node import Node

def Sequence(data, max_length=None):
    var = T.make_sequence(data.get_data(), max_length)
    data = data.next(var, data.get_shape_out())
    data.sequence = True
    data.sequence_length = max_length
    return data

class Generate(Node):

    def __init__(self, node, length=None):
        super(Generate, self).__init__()
        self.node = node
        self.length = length
        assert self.node.get_shape_out() == self.node.get_shape_in()

    def forward(self, X, **kwargs):
        states = [self.node.get_initial_states(X)]
        def step(input, softmax):
            state = states[0]
            out, state = self.node.step(X.next(input, self.get_shape_in()), state)
            out_softmax = out.get_data()
            states[0] = state
            out_sample = T.sample(out_softmax)
            return out_sample, out_softmax

        output, self.updates = T.generate(step, [X.get_data(), X.get_data()], self.length)
        output = X.next(output[1], self.get_shape_out())
        output.sequence = True
        output.sequence_length = self.length
        return output

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

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        if self.stateful:
            N = self.get_batch_size()
        else:
            N = T.shape(X)[1]
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
