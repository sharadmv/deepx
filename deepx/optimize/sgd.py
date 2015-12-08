import theano.tensor as T
from optimizer import Optimizer

class sgd(Optimizer):

    def get_aux_inputs(self):
        return [T.fscalar('training_rate')]

    def updates(self, training_rate):
        return [(p, p - training_rate * g) for p, g in zip(self.parameters, self.grads)]
