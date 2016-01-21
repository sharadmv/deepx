from .. import backend as T

class Optimizer(object):

    def __init__(self, model, loss, clip_gradients=None):
        self.model = model
        self.loss = loss

        self.parameters = self.model.get_parameters()
        self.initialize()

        aux_inputs = self.get_aux_inputs()
        inputs = self.model.get_formatted_input()
        ypred = self.model.get_activation(use_dropout=True)
        y = T.placeholder(shape=ypred.get_data())

        opt_inputs = inputs + [y] + aux_inputs
        opt_output = self.loss.loss(ypred, y)

        self.grads = T.gradients(opt_output, self.parameters)
        if clip_gradients is not None:
            c = abs(clip_gradients)
            self.grads = [T.clip(g, -c, c) for g in self.grads]
        updates = self.updates(*aux_inputs) + self.model.get_updates()
        self.train = T.function(opt_inputs, [opt_output], updates=updates)

    def initialize(self):
        pass

    def init_parameter(self, value):
        param = T.variable(value)
        return param

    def get_inputs(self):
        return self.loss.inputs + self.aux_inputs

    def get_aux_inputs(self):
        raise NotImplementedError

    def get_updates(self):
        return self.updates(*self.aux_inputs)

    def get_result(self):
        return self.model.get_mixin('loss').result
