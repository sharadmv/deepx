from functools import reduce
import copy

from deepx.core import Unary, Op

class Residual(Unary):

    def combinator(self, inputs, outputs):
        return [a + b for a, b in zip(inputs, outputs)]

def Repeat(op, num_repeats):
    ops = [copy.deepcopy(op) for _ in range(num_repeats)]
    return reduce(lambda a, b: a >> b, ops)
