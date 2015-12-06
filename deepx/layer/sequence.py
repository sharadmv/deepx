import theano.tensor as T
import numpy as np
from exceptions import DimensionException
from layer import Layer, CompositeLayer

class RecurrentLayer(Layer):

    def step(self, X, H):
        raise NotImplementedError

    def get_defaults(self):
        return (T.matrix(),)

    def __rshift__(self, layer):
        return CompositeRecurrentLayer(self, layer)

class StatefulRecurrentLayer(RecurrentLayer):

    def get_defaults(self):
        return (T.matrix(), T.matrix())

    def get_initial(self, batch_size):
        return (np.zeros((batch_size, self.n_out)), np.zeros((batch_size, self.n_out)))

class CompositeRecurrentLayer(CompositeLayer, RecurrentLayer):

    def __init__(self, in_layer, out_layer):
        self.in_layer = in_layer
        self.out_layer = out_layer

        if self.in_layer.n_out != self.out_layer.n_in:
            raise DimensionException(self.in_layer.n_out, self.out_layer.n_in)

        self.n_in = self.in_layer.n_in
        self.n_out = self.out_layer.n_out

    def get_defaults(self):
        in_defaults = self.in_layer.get_defaults()
        out_defaults = self.out_layer.get_defaults()
        if not self.in_layer.is_composite():
            in_defaults = [in_defaults]
        if not self.out_layer.is_composite():
            out_defaults = [out_defaults]
        return in_defaults + out_defaults

    def get_initial(self, batch_size):
        in_initial = self.in_layer.get_initial(batch_size)
        out_initial = self.out_layer.get_initial(batch_size)
        if not self.in_layer.is_composite():
            in_initial = [in_initial]
        if not self.out_layer.is_composite():
            out_initial = [out_initial]
        return in_initial + out_initial


    def step(self, X, H):
        assert len(H) == len(self), "Incorrect dimensions"
        (H_in, H_out) = H[:len(self.in_layer)], H[len(self.in_layer):]
        if self.in_layer.is_composite():
            X, H_in = self.in_layer.step(X, H_in)
        else:
            X, H_in = self.in_layer.step(X, H_in[0])
            H_in = [H_in]
        if self.out_layer.is_composite():
            X, H_out = self.out_layer.step(X, H_out)
        else:
            X, H_out = self.out_layer.step(X, H_out[0])
            H_out = [H_out]
        return X, H_in + H_out
