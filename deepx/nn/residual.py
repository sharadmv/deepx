from ..core import FunctionalNode
from .ops import Repeat

class Residual(FunctionalNode):

    def func(self, inputs):
        return inputs[0] + (inputs[0] >> self.node)

    def get_shapes_out(self):
        return self.node.get_shapes_out()

    def get_num_inputs(self):
        return 1

    def get_num_outputs(self):
        return 1

    def __str__(self):
        return "Residual(%s)" % self.node

def RepeatResidual(node, amount):
    return Repeat(Residual(node), amount)
