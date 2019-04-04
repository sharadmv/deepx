from deepx.backend import T
from deepx.core import Layer

class KerasLayer(Layer):

    def __init__(self, layer):
        super(KerasLayer, self).__init__()
        self.layer = layer

    def shape_inference(self):
        if self.get_shape_in() is not None:
            placeholder = T.ones(self.get_shape_in()[0])
            output = self.layer(placeholder)
            self.set_shape_out([T.get_shape(output)])

    def is_initialized(self):
        return True

    def get_parameters(self):
        return self.layer.trainable_weights

    def _forward(self, X):
        return self.layer(X)
