import numpy as np
from .. import backend as T
from ..core import RecurrentLayer, Data
from ..util import pack_tuple, unpack_tuple

class Generate(RecurrentLayer):

    def __init__(self, node, max_length=None, sharpen=np.array(10000).astype(np.float32)):
        self.node = node
        self._sample = None
        self.sharpen_amount = sharpen
        self._cache = None
        self.max_length = max_length
        if max_length is not None:
            self.length = max_length
        else:
            self.length = Data.from_placeholder(T.placeholder(name='generate_length', ndim=0, dtype='int32'), (), None)
        assert self.node.get_shape_out() == self.node.get_shape_in()
        super(Generate, self).__init__(shape_in=self.get_shape_in(), shape_out=self.get_shape_out(), sharpen=sharpen)

    def forward(self, *inputs, **kwargs):
        X = inputs[0]
        if self.max_length is not None:
            length = self.max_length
        else:
            length = inputs[1].get_placeholder()
        output = self.generate(X, length, **kwargs)
        out = Data.from_placeholder(output[0], self.get_shape_out(),
                   batch_size=X.batch_size,
                   sequence=True)
        return [out]

    def is_input(self):
        return True

    def sample(self, X):
        if self._sample is None:
            inputs = self.get_inputs()
            input = inputs[0]
            if self.max_length is not None:
                length = self.max_length
            else:
                length = inputs[1].get_placeholder()
            output = self.generate(input, length, dropout=False)[0]
            self._sample = T.function(self.get_network_inputs(), [output], updates=self.get_updates())
        return self._sample(X)

    def sharpen(self, x, idx):
        sharpened = x + idx * self.sharpen_amount
        return sharpened / T.sum(sharpened, axis=1)[:, None]

    def generate(self, X, length, **kwargs):
        if self._cache is not None:
            return self._cache
        states = self.node.get_initial_states(X.get_placeholder(), shape_index=0)
        states, shape = unpack_tuple(states)

        batch_size = X.batch_size

        def step(input, _, *states):
            packed_state = pack_tuple(states, shape)
            output_softmax, next_state = self.node.step(Data.from_placeholder(input, self.get_shape_out(),
                                                                              batch_size), packed_state)
            output_softmax = output_softmax.get_placeholder()
            output_sample = T.sample(output_softmax)
            output_softmax = self.sharpen(output_softmax, output_sample)
            states, _ = unpack_tuple(next_state)
            return [output_sample, output_softmax] + list(states)

        output, updates = T.generate(step, [X.get_placeholder(), X.get_placeholder()] + list(states), length)
        if not len(self.updates):
            self.updates = updates
        self._cache = output
        return output

    def can_initialize(self):
        return True

    def infer_shape(self):
        self.node.infer_shape()

    def get_inputs(self):
        if self.max_length is not None:
            return self.node.get_inputs()
        return self.node.get_inputs() + [self.length]

    def get_network_inputs(self):
        return self.node.get_network_inputs() + [self.length.get_placeholder()]

    def get_shape_in(self):
        return self.node.get_shape_in()

    def get_shape_out(self):
        return self.node.get_shape_out()

    def tie(self, node):
        return Generate(self.node.tie(node), max_length=self.max_length)

    def get_state(self):
        return self.node.get_state()

    def set_state(self, state):
        self.node.set_state(state)

    def get_parameters(self):
        return self.node.get_parameters()

    def __str__(self):
        return "Generate(%s)" % str(self.node)

