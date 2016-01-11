from .node import Node, Data

class RecurrentNode(Node):

    def recurrent_forward(self, X):
        return Data(self._forward(X.get_data()), self.get_shape_out(), sequence=True)

    def is_recurrent(self):
        return True
