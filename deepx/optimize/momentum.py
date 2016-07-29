from .. import T
from .optimizer import Optimizer

class Momentum(Optimizer):

    def initialize(self):
        self.caches = [T.variable(T.get_value(p) * 0) for p in self.parameters]

    def get_aux_inputs(self):
        return [
            T.placeholder(ndim=0, name='rho'),
            T.placeholder(ndim=0, name='eta')
        ]

    def updates(self, rho, eta):
        updates = []
        for p, c, g in zip(self.parameters, self.caches, self.grads):
            delta = rho * g + (1 - rho) * c
            updates.append((c, delta))
            updates.append((p, p - eta * delta))
        return updates
