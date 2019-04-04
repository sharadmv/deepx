from deepx.backend import T
from deepx.core import Layer

class KerasLayer(Layer):

    name_counter = 0

    def __init__(self, layer):
        super(KerasLayer, self).__init__()
        self.layer = layer
        def setstate(this, state):
            print(state)
            super(self.layer.__class__, this).__setstate__(state)
        self.layer.__setstate__ = setstate

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.layer._name = self.layer.name + "-" + str(KerasLayer.name_counter)
        KerasLayer.name_counter += 1

    def shape_inference(self):
        return

    def is_initialized(self):
        return True

    def get_parameters(self):
        return self.layer.trainable_weights

    def _forward(self, X):
        return self.layer(X)
