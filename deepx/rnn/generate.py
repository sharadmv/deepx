from .. import backend as T
from ..node import Node
from ..util import pack_tuple, unpack_tuple

class Generate(Node):

    def __init__(self, node, length=None):
        super(Generate, self).__init__()
        self.node = node
        self.length = length
        assert self.node.get_shape_out() == self.node.get_shape_in()

    def forward(self, X, **kwargs):
        states = self.node.get_initial_states(X.get_data(), shape_index=0)
        states, shape = unpack_tuple(states)

        def step(input, *states):
            packed_state = pack_tuple(states, shape)
            output_softmax, next_state = self.node.step(X.next(input), packed_state)
            output_softmax = output_softmax.get_data()
            output_sample = T.sample(output_softmax)
            states, _ = unpack_tuple(next_state)
            return [output_sample] + list(states)

        output, self.updates = T.generate(step, [X.get_data()] + list(states), self.length)
        out = X.next(output[0], self.get_shape_out())
        out.sequence = True
        out.sequence_length = self.length
        return out

    def can_initialize(self):
        return True

    def infer_shape(self):
        self.node.infer_shape()

    def get_input(self):
        return self.node.get_input()

    def get_shape_in(self):
        return self.node.get_shape_in()

    def get_shape_out(self):
        return self.node.get_shape_out()

    def tie(self, node):
        return Generate(self.node.tie(node), length=self.length)

    def get_state(self):
        return self.node.get_state()

    def set_state(self, state):
        self.node.set_state(state)

    def get_parameters(self):
        return self.node.get_parameters()

    def __str__(self):
        return "Generate(%s)" % str(self.node)

