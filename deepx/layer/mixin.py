from theanify import Theanifiable, theanify, DecoratedTheano

from model import Data
from exceptions import DataException

class Mixin(Theanifiable):
    name = 'mixin'
    priority = 1

    def __init__(self):
        super(Mixin, self).__init__()

    def setup(self, model):
        self.model = model
        self.in_var = model.get_in_var()
        self.input_vars = []
        self.layer_vars = []
        self.aux_vars = self.get_aux_vars()
        for layer in model:
            layer_var = layer.get_layer_var()
            if layer.is_recurrent():
                self.input_vars.append(layer_var)
            self.layer_vars.append(layer_var)
        mix_args = [self.in_var] + self.input_vars + self.aux_vars
        self.mix = DecoratedTheano(self.mix, mix_args)
        self.mix.set_instance(self)

    def mix(model, self, X, *args):
        data = Data(X, self.layer_vars)
        activations = data > self.model
        return self.mixin(activations, *args)

    def get_aux_vars(self):
        raise NotImplementedError

class OutputMixin(Mixin):

    name = 'output'

    def get_aux_vars(self):
        return []

    def mixin(self, activations, *args):
        return activations[-1].get_data()

output = OutputMixin()
