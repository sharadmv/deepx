from deepx.backend import T
from deepx.core import Layer

class Dropout(Layer):

    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        assert 0 <= self.p < 1, 'Invalid dropout value'

    def initialize(self):
        return

    def is_initialized(self):
        return True

    def shape_inference(self):
        if self.get_shape_in() is not None:
            self.set_shape_out(self.get_shape_in())

    def _forward(self, X, **kwargs):
        if T.get_context('train'):
            return T.dropout(X, self.p)
        else:
            return X
