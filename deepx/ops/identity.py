from deepx.core import Op

__all__ = ["Identity"]

class Identity(Op):

    def forward(self, *inputs):
        return inputs

    def shape_inference(self):
        self.set_shape_out(self.get_shape_in())

    def is_initialized(self):
        return True
