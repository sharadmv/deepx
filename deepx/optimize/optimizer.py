from .. import T

class Optimizer(object):

    def __init__(self, network, clip_gradients=None):
        self.network = network

        self.initialize()

        self.optimization_parameters = self.get_optimization_parameters()
        self.network_output = self.network.outputs()[0]

        self.gradients = T.gradients(self.network_outputs[0], self.optimization_parameters)

        if clip_gradients is not None:
            c = abs(clip_gradients)
            self.grads = [self.scale(g, c) for g in self.grads]

        self.grad_updates = self.get_updates() + self.loss.get_graph_updates()
        self.opt_inputs = self.loss.get_graph_inputs()

        self._gradient = None
        self._loss = None
        self._train = None

    def get_gradient(self):
        return T.gradients(self.opt_outputs[0], self.parameters)

    def gradient(self, *args):
        if self._gradient is None:
            self._gradient = T.function(self.opt_inputs, self.grads)
        return self._gradient(*args)

    def batch_loss(self, *args):
        if self._loss is None:
            self._loss = T.function(self.opt_inputs, self.opt_outputs, updates=self.loss.get_updates())
        return self._loss(*args)

    def train(self, *args):
        if self._train is None:
            self._train = T.function(self.opt_inputs + self.aux_inputs, self.opt_outputs, updates=self.grad_updates)
        return self._train(*args)

    def clip(self, X, epsilon):
        return T.maximum(T.minimum(X, epsilon), -epsilon)

    def scale(self, X, max_norm):
        curr_norm = T.sum(T.abs(X))
        return T.ifelse(T.lt(curr_norm, max_norm), X, max_norm * (X / curr_norm))

    def reset_parameters(self):
        pass

    def initialize(self):
        pass

    def init_parameter(self, value):
        param = T.variable(value)
        return param

    def get_aux_inputs(self):
        raise NotImplementedError

    def get_updates(self):
        return self.updates(*self.aux_inputs)
