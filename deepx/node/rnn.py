from .node import Node, Data

class RecurrentNode(Node):

    def forward(self, X, **kwargs):
        if self.stateful:
            self.batch_size = X.batch_size
        return super(RecurrentNode, self).forward(X, **kwargs)

    def recurrent_forward(self, X):
        return Data(self._forward(X.get_data()), self.get_shape_out(), sequence=True,
                    batch_size=X.batch_size)

    def is_recurrent(self):
        return True
