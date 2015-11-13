import theano
import theano.tensor as T
from optimizer import Optimizer

class RMSProp(Optimizer):

    def __init__(self, parameter_model):

        self.average_gradient = [theano.shared(p.get_value() * 0) for p in parameter_model.get_parameters()]
        self.average_rms = [theano.shared(p.get_value() * 0) for p in parameter_model.get_parameters()]
        self.parameter_update = [theano.shared(p.get_value() * 0) for p in parameter_model.get_parameters()]

        super(RMSProp, self).__init__(parameter_model, [
            T.scalar('learning_rate')
        ])


    def updates(self, learning_rate, *args):
        grads = self.grads
        next_average_gradient = [0.95 * avg + 0.05 * g for g, avg in zip(grads, self.average_gradient)]
        next_rms = [0.95 * rms + 0.05 * (g ** 2) for g, rms in zip(grads, self.average_rms)]
        next_parameter = [0.9 * param_update - 1e-4 * g / T.sqrt(rms - avg ** 2 + 1e-4)
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

        updates = [(p, p + learning_rate * param_update) for p, param_update in zip(self.get_parameters(), next_parameter)]

        return updates + average_gradient_update + rms_update + next_parameter_update

