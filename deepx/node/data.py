from .. import backend as T
from .node import Node

class Data(Node):

    def __init__(self, data, shape=None, sequence=False, batch_size=None, sequence_length=None):
        super(Data, self).__init__()
        self.data = data
        self.shape_in = shape
        self.shape_out = shape

        self.batch_size = batch_size

        self.sequence = sequence
        self.sequence_length = sequence_length


    def _infer(self, shape_in):
        return self.get_shape_out()

    def can_initialize(self):
        return True

    def initialize(self):
        return

    def _forward(self, X, **kwargs):
        return X

    def __getitem__(self, idx):
        return self.index(idx)

    def index(self, idx):
        return Data(self.data[idx], self.shape_out, batch_size=self.batch_size)

    @property
    def ndim(self):
        return T.ndim(self.data)

    def get_input(self):
        return self

    def get_data(self):
        return self.data

    def is_data(self):
        return True

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.sequence:
            return "Sequence(%s, %s)" % (self.data, self.get_shape_out())
        return "Data(%s, %s)" % (self.data, self.get_shape_out())

    def next(self, data, shape=None):
        return Data(data, shape=shape, sequence=self.sequence, batch_size=self.batch_size,
                    sequence_length=self.sequence_length)

    def copy(self, **kwargs):
        return Data(self.data, shape=self.get_shape_out(), sequence=self.sequence, batch_size=self.batch_size,
                    sequence_length=self.sequence_length)

    def set_state(self, state):
        pass

    def is_sequence(self):
        return self.sequence
