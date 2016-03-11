from .. import backend as T
from .optimizer import Optimizer

class RMSProp(Optimizer):

    def initialize(self):
        self.average_gradient = [T.variable(T.get_value(p) * 0) for p in self.parameters]
        self.average_rms = [T.variable(T.get_value(p) * 0) for p in self.parameters]
        self.parameter_update = [T.variable(T.get_value(p) * 0) for p in self.parameters]

    def reset_parameters(self):
        for param in [self.average_gradient, self.average_rms, self.parameter_update]:
            for p in param:
                T.set_value(p, T.get_value(p) * 0)

    def get_aux_inputs(self):
        return [T.placeholder(ndim=0, name='learning_rate')]

    def updates(self, learning_rate):
        grads = self.grads
        next_average_gradient = [0.95 * avg + 0.05 * g for g, avg in zip(grads, self.average_gradient)]
        next_rms = [0.95 * rms + 0.05 * T.pow(g, 2) for g, rms in zip(grads, self.average_rms)]
        next_parameter = [0.9 * param_update - 1e-4 * g / T.sqrt(rms - T.pow(avg, 2) + 1e-4)
                        for g, avg, rms, param_update in zip(grads,
                                                            self.average_gradient,
                                                            self.average_rms,
                                                            self.parameter_update)]

        average_gradient_update = [(avg, next_avg) for avg, next_avg in zip(self.average_gradient,
                                                                            next_average_gradient)]
        rms_update = [(rms, rms2) for rms, rms2 in zip(self.average_rms,
                                                            next_rms)]
        next_parameter_update = [(param, param_update) for param, param_update in zip(self.parameter_update,
                                                                                    next_parameter)]

        updates = [(p, p + learning_rate * param_update) for p, param_update in zip(self.parameters, next_parameter)]

        return updates + average_gradient_update + rms_update + next_parameter_update
