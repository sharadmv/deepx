import theano.tensor as T
from optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, parameter_model):
        super(SGD, self).__init__(parameter_model, optimize_args=[
            T.fscalar('training_rate')
        ])


    def updates(self, training_rate, *args):
        return [(p, p - training_rate * g) for p, g in zip(self.get_parameters(), self.grads)]
