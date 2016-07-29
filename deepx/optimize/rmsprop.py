from .. import T
from .optimizer import Optimizer

class RMSProp(Optimizer):

    def __init__(self, loss, rho=0.9, epsilon=1e-6, clip_gradients=None):
        self.rho = rho
        self.epsilon = epsilon
        super(RMSProp, self).__init__(loss, clip_gradients=clip_gradients)

    def initialize(self):
        self.average_gradient = [T.variable(T.zeros_like(p)) for p in self.parameters]

    def reset_parameters(self):
        for p in self.average_gradient:
            T.set_value(p, T.get_value(p) * 0)

    def get_aux_inputs(self):
        return [T.scalar(name='learning_rate')]

    def updates(self, learning_rate):
        updates = []

        for param, grad, avg in zip(self.parameters, self.grads, self.average_gradient):

            next_avg = self.rho * avg + (1 - self.rho) * T.square(grad)
            next_param = param - learning_rate * grad / T.sqrt(next_avg + self.epsilon)

            updates.append((avg, next_avg))
            updates.append((param, next_param))

        return updates
