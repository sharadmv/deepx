from .. import T
from .optimizer import Optimizer

class Adam(Optimizer):

    def __init__(self, loss, clip_gradients=None, b1=0.9, b2=0.999, epsilon=1e-8):
        self.b1_init = b1
        self.b2_init = b2
        self.epsilon = epsilon
        super(Adam, self).__init__(loss, clip_gradients=clip_gradients)

    def initialize(self):
        self.i = T.variable(0.0)
        self.b1 = T.variable(self.b1_init, name='beta1')
        self.b2 = T.variable(self.b2_init, name='beta2')
        self.ms = [T.variable(T.zeros_like(p)) for p in self.parameters]
        self.vs = [T.variable(T.zeros_like(p)) for p in self.parameters]

    def reset_parameters(self):
        for param in [self.ms, self.vs]:
            for p in param:
                T.set_value(p, T.get_value(p) * 0)

    def get_aux_inputs(self):
        return [T.scalar(name='learning_rate')]

    def updates(self, learning_rate):

        updates = []
        t = self.i + 1
        lr_t = learning_rate * T.sqrt(1 - T.pow(self.b2, t)) / (1 - T.pow(self.b1, t))

        for p, g, m, v in zip(self.parameters, self.grads, self.ms, self.vs):

            m_t = (self.b1 * m) + (1 - self.b1) * g
            v_t = (self.b2 * v) + (1 - self.b2) * T.square(g)
            p_t = p - lr_t * m_t / (T.sqrt(v_t) + self.epsilon)

            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t))
        updates.append((self.i, self.i + 1))
        return updates
