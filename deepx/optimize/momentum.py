import theano
import theano.tensor as T
from optimizer import Optimizer

class Momentum(Optimizer):

    def __init__(self, parameter_model):
        self.caches = [theano.shared(p.get_value() * 0) for p in parameter_model.get_parameters()]

        super(Momentum, self).__init__(parameter_model, optimize_args=[
            T.fscalar('rho'),
            T.fscalar('eta'),
        ])

    def updates(self, rho, eta, *args):
        updates = []
        for p, c, g in zip(self.get_parameters(), self.caches, self.grads):
            delta = rho * g + (1 - rho) * c
            updates.append((c, delta))
            updates.append((p, p - eta * delta))
        return updates
