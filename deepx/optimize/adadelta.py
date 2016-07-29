from .. import T
from .optimizer import Optimizer

class Adadelta(Optimizer):

    def __init__(self, loss, clip_gradients=None, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon
        super(Adadelta, self).__init__(loss, clip_gradients=clip_gradients)

    def initialize(self):
        self.a = [T.variable(T.zeros_like(p)) for p in self.parameters]
        self.d = [T.variable(T.zeros_like(p)) for p in self.parameters]

    def reset_parameters(self):
        for param in [self.a, self.d]:
            for p in param:
                T.set_value(p, T.get_value(p) * 0)

    def get_aux_inputs(self):
        return [T.placeholder(ndim=0, name='learning_rate')]

    def updates(self, learning_rate):

        updates = []
        for p, g, a, d in zip(self.parameters, self.grads, self.a,
                                   self.d):
            new_a = self.rho * a + (1 - self.rho) * T.square(g)
            updates.append((a, new_a))

            update = g * T.sqrt(d + self.epsilon) / T.sqrt(new_a + self.epsilon)

            new_p = p - learning_rate * update
            updates.append((p, new_p))

            new_d_a = self.rho * d + (1 - self.rho) * T.square(update)
            updates.append((d, new_d_a))
        return updates
