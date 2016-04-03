from .. import backend as T
from ..core import Layer

class Dropout(Layer):

    def __init__(self, p=0.5, *args, **kwargs):
        super(Dropout, self).__init__(p=p, *args, **kwargs)
        self.p = self.config.get('p', 0.5)
        assert 0 <= self.p < 1, 'Invalid dropout value'

    def initialize(self):
        pass

    def _infer(self, shape_in):
        return shape_in

    def forward(self, *args, **kwargs):
        return super(Dropout, self).forward(ignore_sequence=True, use_kwargs=True, *args, **kwargs)

    def _forward(self, X, dropout=True):
        if dropout:
            return T.dropout(X, self.p)
        else:
            return X
