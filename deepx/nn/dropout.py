from .. import backend as T
from ..node import Node, Data

class Dropout(Node):

    def __init__(self, p, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)
        self.p = p
        assert 0 <= p < 1, 'Invalid dropout value'

    def can_initialize(self):
        return True

    def _infer(self, shape_in):
        return shape_in

    def forward(self, X, use_dropout=True):
        return X.next(self._forward(X.get_data(), use_dropout=use_dropout), self.get_shape_out())

    def _forward(self, X, use_dropout=True):
        if use_dropout:
            return T.dropout(X, self.p)
        else:
            return X

    def copy(self, **kwargs):
        return Dropout(self.p)
