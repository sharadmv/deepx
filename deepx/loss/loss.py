from abc import abstractmethod

from .. import T
from ..core import ShapedNode, Data, Shape

class Loss(ShapedNode):

    def __init__(self):
        super(Loss, self).__init__(None, [Shape(())])
        self.y = None

    def inputs(self):
        if self.get_num_inputs() > 1:
            return []
        if self.y is None:
            self.y = self.get_shapes_in()[0].create_placeholder()
        return [self.y]

    def outputs(self, *inputs):
        if self.get_num_inputs() > 1:
            ypred, y = inputs
        else:
            ypred, y = inputs[0], self.y
        return self.loss(ypred, y)

    # Shape inference

    def infer_shape(self):
        pass

    @abstractmethod
    def loss(self, ypred, y):
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s()" % self.__class__.__name__
