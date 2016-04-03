import numpy as np
from .. import backend as T
from ..core import RecurrentLayer
from ..util import pack_tuple, unpack_tuple

class Generate(RecurrentLayer):

    def __init__(self, node, length=None, sharpen=np.array(10000).astype(np.float32)):
        super(Generate, self).__init__(node, length=length, sharpen=sharpen)
        self.node = node
        self.length = length
        self._sample = None
        self.updates = None
        self.sharpen_amount = sharpen
        self._cache = None
        assert self.node.get_shape_out() == self.node.get_shape_in()

    def forward(self, X, **kwargs):
        output = self.generate(X, **kwargs)
        out = X.next(output[1], self.get_shape_out())
        out.sequence = True
        out.sequence_length = self.length
        return out

    def sample(self, X):
        if self._sample is None:
            input = self.get_input()
            output = self.generate(input, use_dropout=False)[0]
            self._sample = T.function(self.get_formatted_input(), [output], updates=self.get_updates())
        return self._sample(X)

    def sharpen(self, x, idx):
        sharpened = x + idx * self.sharpen_amount
        return sharpened / T.sum(sharpened, axis=1)[:, None]

    def generate(self, X, **kwargs):
        if self._cache is not None:
            return self._cache
        states = self.node.get_initial_states(X.get_data(), shape_index=0)
        states, shape = unpack_tuple(states)

        def step(input, _, *states):
            packed_state = pack_tuple(states, shape)
            output_softmax, next_state = self.node.step(X.next(input), packed_state)
            output_softmax = output_softmax.get_data()
            output_sample = T.sample(output_softmax)
            output_softmax = self.sharpen(output_softmax, output_sample)
            states, _ = unpack_tuple(next_state)
            return [output_sample, output_softmax] + list(states)

        output, updates = T.generate(step, [X.get_data(), X.get_data()] + list(states), self.length)
        if self.updates is None:
            self.updates = updates
        self._cache = output
        return output

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

