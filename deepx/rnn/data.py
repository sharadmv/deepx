from .. import backend as T
from ..node import Data

class Sequence(Data):

    def __init__(self, data_var, max_length=None):
        self.sequence_dim = data_var.ndim
        self.max_length = max_length

        data = T.make_sequence(data_var.get_data(), self.max_length)
        super(Sequence, self).__init__(data)
        self.shape_in = data_var.shape_in
        self.shape_out = data_var.shape_out
        self._is_sequence = True

    def __str__(self):
        return "Sequence(%s)" % str(self.get_shape_out())

    def is_sequence(self):
        return self._is_sequence
