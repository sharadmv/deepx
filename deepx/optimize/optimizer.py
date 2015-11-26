import numpy as np
import theano.tensor as T

from theanify import theanify, Theanifiable

class Optimizable(Theanifiable):

    def __init__(self):
        super(Optimizable, self).__init__()

    def cost(self, *args, **kwargs):
        raise NotImplementedError


class Optimizer(Theanifiable):

    def __init__(self, optimizable, optimize_args=[], clip=5):
        super(Optimizer, self).__init__()
        self.model = optimizable
        self.cost_function = self.model.cost
        self.cost_updates = []

        if not hasattr(self.cost_function, 'args'):
            raise Exception('Please annotate cost with @theanify')

        self.cost_args = self.model.cost.args
        self.optimize_args = optimize_args

        self.cost_result = self.model.cost(*self.cost_args)
        if self.cost_function.returns_updates:
            self.cost, cost_updates = self.cost_result
            self.cost_updates.extend(cost_updates)
        else:
            self.cost = self.cost_result

        self.rest = ()
        if isinstance(self.cost, tuple):
            self.rest = self.cost[1:]
            self.cost = self.cost[0]
        self.grads = T.grad(self.cost, self.get_parameters())

        self.compile_method('optimize', args=self.optimize_args + list(self.cost_args))

    def get_initial_state(self, batch_size):
        return np.zeros((batch_size, self.model.n_layers, self.model.n_hidden))

    def train(self, *args):
        cost_args = args[:len(self.cost_args)]
        training_args = args[len(self.cost_args):]
        return self.optimize(*(training_args + cost_args))

    @theanify(updates="updates", returns_updates=True)
    def optimize(self, *args):
        if len(self.rest):
            return (self.cost,) + self.rest, self.cost_updates
        return self.cost, self.cost_updates

    def get_parameters(self):
        return self.model.get_parameters()
