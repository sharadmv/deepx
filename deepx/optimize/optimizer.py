import theano
import theano.tensor as T

from ..node import Mixin

class Optimizer(Mixin):

    name = 'train'
    priority = 1

    def init_parameters(self):
        pass

    def setup(self, model):
        self.model = model
        assert model.has_mixin('loss')

        self.arch = model.arch

        self.parameters = self.arch.get_parameters()
        self.init_parameters()

        self.loss = self.model.get_mixin('loss')
        self.grads = T.grad(self.loss.result, self.parameters)

        self.aux_inputs = self.get_aux_inputs()

        self.inputs = self.get_inputs()
        self.result = self.get_result()
        self.gradient_updates = self.updates(*self.aux_inputs)

        self.func = self.create_function()

    def get_inputs(self):
        return self.loss.inputs + self.aux_inputs

    def get_aux_inputs(self):
        raise NotImplementedError

    def get_result(self):
        return self.model.get_mixin('loss').result

    def create_function(self):
        return theano.function(self.inputs, self.result, updates=self.gradient_updates, allow_input_downcast=True)
