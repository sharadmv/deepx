from .. import T
from .optimizer import Optimizer

class SGD(Optimizer):

    def get_aux_inputs(self):
        return [T.placeholder(ndim=0, name='training_rate')]

    def updates(self, training_rate):
        return [(p, p - training_rate * g) for p, g in zip(self.parameters, self.grads)]
