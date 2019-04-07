from abc import abstractmethod

from deepx.core.op import Op

class Layer(Op):

    def __init__(self, shape_in=None, shape_out=None):
        super(Layer, self).__init__()

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def forward(self, *inputs, **kwargs):
        assert len(inputs) == 1
        output = self._forward(inputs[0], **kwargs)
        return [output]

    @abstractmethod
    def _forward(self, X):
        pass

class ShapedLayer(Layer):

    def __init__(self, shape_in=None, shape_out=None):
        super(ShapedLayer, self).__init__()
        if shape_out is not None:
            if not isinstance(shape_in, list):
                shape_in = [shape_in]
            if not isinstance(shape_out, list):
                shape_out = [shape_out]
            self.dim_in, self.dim_out = shape_in, shape_out
        elif shape_in is not None and shape_out is None:
            if not isinstance(shape_in, list):
                shape_in = [shape_in]
            self.dim_in, self.dim_out = None, shape_in
        else:
            self.dim_in = self.dim_out = None
        self.initialized = False

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def shape_inference(self):
        shape_in = self.get_shape_in()
        dim_in, dim_out = self.get_dim_in(), self.get_dim_out()
        if shape_in is not None:
            assert len(shape_in) == 1, "Cannot pass multiple inputs into layer"
            shape_in = shape_in[0]
            if dim_in is not None:
                if dim_in != shape_in[-len(dim_in):]:
                    raise Exception("Shape inference error")
            self.dim_in = shape_in[-1:]
            assert dim_out is not None
            self.set_shape_out([shape_in[:len(dim_out)] + dim_out])
        else:
            if dim_in is not None:
                self.set_shape_in([[None] + dim_in])
        if self.get_shape_in() is not None:
            self.set_shape_out([self.get_shape_in()[0][:-1] + dim_out])
        if not self.initialized and self.get_shape_in() is not None and self.get_shape_out() is not None:
            self.initialize()
            self.initialized = True

    def is_initialized(self):
        return self.initialized

    def __repr__(self):
        name = self.__class__.__name__
        return "%s(%s, %s)" % (name, self.dim_in, self.dim_out)
