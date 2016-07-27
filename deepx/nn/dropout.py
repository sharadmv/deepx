from .. import T
from ..core import Layer

class Dropout(Layer):

    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        assert 0 <= self.p < 1, 'Invalid dropout value'

    def initialize(self):
        return

    def infer(self, shape_in):
        return shape_in

    def forward(self, X, dropout=True):
        if dropout:
            return T.dropout(X, self.p)
        else:
            return X
