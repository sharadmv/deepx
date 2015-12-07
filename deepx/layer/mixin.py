from theanify import Theanifiable, theanify, DecoratedTheano

from model import Data

class Mixin(Theanifiable):
    name = 'mixin'
    priority = 1

    def __init__(self):
        super(Mixin, self).__init__()

    def get_aux_vars(self, model):
        raise NotImplementedError

    def get_input_vars(self, model):
        raise NotImplementedError

    def setup(self, model):
        self.in_vars = self.get_input_vars(model)
        self.aux_vars = self.get_vars(model)

        self.inputs = self.in_vars + self.aux_vars

    def mix(model, self, X, *args):
        data = Data(X, self.layer_vars)
        activations = data > self.model
        return self.mixin(activations, *args)

class OutputMixin(Mixin):

    name = 'output'

    def get_input_vars(self, model):
        return []

    def get_aux_vars(self, model):
        return []

    def mixin(self, activations, *args):
        return activations[-1].get_data()

output = OutputMixin()
