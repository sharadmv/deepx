from deepx.core import Layer

class KerasLayer(Layer):

    def __init__(self, layer):
        super(KerasLayer, self).__init__()
        self.layer = layer

    def shape_inference(self):
        pass

    def is_initialized(self):
        return True

    def get_parameters(self):
        return self.layer.trainable_weights

    def get_shape_out(self):
        try:
            return self.layer.output_shape
        except:
            return None

    def get_shape_in(self):
        try:
            return self.layer.input_shape
        except:
            return None


    def _forward(self, X):
        return self.layer(X)
