import theano
import theano.tensor as T
from optimizer import Optimizer

class momentum(Optimizer):

    def init_parameters(self):
        self.caches = [theano.shared(p.get_value() * 0) for p in self.parameters]

    def get_aux_inputs(self):
        return [
            T.fscalar('rho'),
            T.fscalar('eta'),
        ]

    def updates(self, rho, eta):
        updates = []
        for p, c, g in zip(self.parameters, self.caches, self.grads):
            delta = rho * g + (1 - rho) * c
            updates.append((c, delta))
            updates.append((p, p - eta * delta))
        return updates
