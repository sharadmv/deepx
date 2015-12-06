import theano.tensor as T
from theanify import DecoratedTheano

from ..layer import Mixin

class Optimizer(Mixin):

    name = 'train'

    priority = -1

    def get_aux_vars(self):
        raise NotImplementedError

    def setup(self, model):
        assert "loss" in model.mixins, "Must add loss mixin before optimizer mixin"
        self.model = model
        loss_mixin = self.model.mixins['loss']
        self.model = model
        self.in_var = loss_mixin.in_var
        self.input_vars = loss_mixin.input_vars
        self.layer_vars = loss_mixin.layer_vars
        self.prev_aux_vars = loss_mixin.aux_vars
        aux_vars = self.get_aux_vars()

        loss_args = [self.in_var] + self.input_vars + self.prev_aux_vars
        self.parameters = self.model.get_parameters()
        self.init_parameters()

        self.loss = self.model.loss(*loss_args)
        self.grads = T.grad(self.loss, self.model.get_parameters())
        self.updates = self.updates(*aux_vars)

        opt_args = loss_args + aux_vars
        self.mix = DecoratedTheano(self.mix, opt_args, updates=self.updates)
        self.mix.set_instance(self)

    def get_opt_vars(self):
        raise NotImplementedError

    def mixin(self, *args):
        return self.loss
