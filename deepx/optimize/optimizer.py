from .. import backend as T

from ..node import Mixin

class Optimizer(Mixin):

    name = 'train'
    priority = 1

    def initialize(self):
        pass

    def init_parameter(self, value):
        param = T.variable(value)
        return param

    def setup(self, model):
        super(Optimizer, self).setup(model)
        assert model.has_mixin('loss')

        self.parameters = self.arch.get_parameters()
        self.initialize()

        self.loss = self.model.get_mixin('loss')
        self.grads = T.gradients(self.loss.result, self.parameters)

        self.aux_inputs = self.get_aux_inputs()

    def get_inputs(self):
        return self.loss.inputs + self.aux_inputs

    def get_aux_inputs(self):
        raise NotImplementedError

    def get_updates(self):
        return self.updates(*self.aux_inputs)

    def get_result(self):
        return self.model.get_mixin('loss').result
