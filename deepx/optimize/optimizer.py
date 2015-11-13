import numpy as np
import theano.tensor as T

from theanify import theanify, Theanifiable

class Optimizer(Theanifiable):

    def __init__(self, parameter_model, optimize_args=[], clip=5):
        super(Optimizer, self).__init__()
        self.parameter_model = parameter_model
        self.cost_args = self.parameter_model.cost.args
        self.optimize_args = optimize_args

        (self.cost, self.state), self.rng_updates = self.parameter_model.cost(*self.cost_args)
        # self.grads = map(lambda x: x.clip(-clip, clip), T.grad(self.cost, self.get_parameters()))
        self.grads = T.grad(self.cost, self.get_parameters())

        self.compile_method('optimize', args=self.optimize_args + list(self.cost_args))

    def get_initial_state(self, batch_size):
        return np.zeros((batch_size, self.parameter_model.n_layers, self.parameter_model.n_hidden))

    def train(self, *args):
        cost_args = args[:len(self.cost_args)]
        training_args = args[len(self.cost_args):]
        return self.optimize(*(training_args + cost_args))

    @theanify(updates="updates", returns_updates=True)
    def optimize(self, *args):
        return (self.cost, self.state), self.rng_updates

    def get_parameters(self):
        return self.parameter_model.get_parameters()
