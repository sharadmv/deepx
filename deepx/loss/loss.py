from abc import abstractmethod

from .. import T
from ..core import Node, Data, Shape

class Loss(Node):

    def __init__(self):
        super(Loss, self).__init__()
        self.y = None
        self.set_shapes_out([
            Shape(())
        ])

    def get_outputs(self, *inputs, **kwargs):
        X = inputs[0]
        if self.y is None:
            y = inputs[1]
        else:
            y = self.y
        if X.is_sequence():
            def step(*inputs):
                return self.loss(*inputs)
            loss = T.scan(step, [X.get_placeholder(), y.get_placeholder()])
            loss = self.sequence_loss(loss)
        else:
            loss = self.loss(X.get_placeholder(), y.get_placeholder())
        return [Data(self.get_shapes_out()[0], placeholder=loss)]

    def get_graph_inputs(self):
        if self.y is None:
            return []
        return [self.y.get_placeholder()]

    def get_graph_parameters(self):
        return []

    def get_graph_updates(self, **kwargs):
        return []

    def reset_states(self):
        return

    def reset_state(self, i):
        return

    def initialize(self, **kwargs):
        return

    def reinitialize(self, **kwargs):
        return

    # Shape inference

    def set_shapes_in(self, shapes_in):
        assert len(shapes_in) == 1 or len(shapes_in) == 2
        self.shapes_in = shapes_in

    def set_shapes_out(self, shapes_out):
        self.shapes_out = shapes_out

    def get_shapes_in(self):
        return self.shapes_in

    def get_shapes_out(self):
        return self.shapes_out

    def get_num_inputs(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is None:
            return None
        return len(shapes_in)

    def get_num_outputs(self):
        return 1

    def infer_shape(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            self.set_shapes_out([shapes_in[0].copy(
                 dim=()
            )])
            if len(shapes_in) == 1:
                self.y = Data(shapes_in[0], name='y')

    def get_state(self):
        return None

    def set_state(self, state):
        return

    def sequence_loss(self, loss):
        return T.mean(loss)

    @abstractmethod
    def loss(self, ypred, y):
        pass

    def __str__(self):
        return "%s()" % self.__class__.__name__
