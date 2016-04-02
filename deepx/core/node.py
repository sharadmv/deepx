from .. import backend as T

class Node(object):

    def __init__(self):

        self._predict = None
        self._predict_dropout = None

        self.frozen = False

    def get_inputs(self):
        raise NotImplementedError

    def get_network_inputs(self):
        return [x.get_placeholder() for x in self.get_inputs()]

    def get_outputs(self, **kwargs):
        raise NotImplementedError

    def get_network_outputs(self, **kwargs):
        return [x.get_placeholder() for x in self.get_outputs(**kwargs)]

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        if not self.is_input():
            raise Exception("Cannot pass through data without an input node.")
        dropout = kwargs.pop('dropout', False)
        if dropout:
            if self._predict_dropout is None:
                self._predict_dropout = T.function(
                    self.get_network_inputs(),
                    self.get_network_outputs(dropout=True)
                )
            return self._predict_dropout(*args, **kwargs)
        else:
            if self._predict is None:
                self._predict = T.function(
                    self.get_network_inputs(),
                    self.get_network_outputs(dropout=False)
                )
            return self._predict(*args, **kwargs)

    def is_input(self):
        raise NotImplementedError

    def has_parameters(self):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def freeze(self):
        node = self.copy(keep_params=True)
        node.frozen = True
        return node

    def unfreeze(self):
        node = self.copy(keep_params=True)
        node.frozen = False
        return node

    # Shape inference

    def get_shape(self):
        return (self.get_shape_in(), self.get_shape_out())

    def set_shape_in(self, shape_in):
        raise NotImplementedError

    def set_shape_out(self, shape_out):
        raise NotImplementedError

    def get_shape_in(self):
        raise NotImplementedError

    def get_shape_out(self):
        raise NotImplementedError

    def is_configured(self):
        raise NotImplementedError

    def infer_shape(self):
        raise NotImplementedError

    # Node functions

    def chain(self, node):
        from binary import Chain
        return Chain(self, node)

    # Infix operation

    def __rshift__(self, node):
        return self.chain(node)

    # Node IO

    def get_options(self):
        return ([], {})

    def get_state(self, **kwargs):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def copy(self, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        return super(Node, self).__repr__()

class ShapedNode(Node):

    def __init__(self, shape_in=None, shape_out=None):
        super(ShapedNode, self).__init__()

        self.set_shape_in(shape_in)
        self.set_shape_out(shape_out)

        self.infer_shape()

    def is_input(self):
        return False

    def get_inputs(self):
        return []

    def get_outputs(self, **kwargs):
        inputs = self.get_inputs()
        return self.forward(*inputs, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


    # Shape inference

    def _infer(self, shape_in):
        raise NotImplementedError

    def set_shape_in(self, shape_in):
        self.shape_in = shape_in

    def set_shape_out(self, shape_out):
        self.shape_out = shape_out

    def get_shape_in(self):
        return self.shape_in

    def get_shape_out(self):
        return self.shape_out

    def is_configured(self):
        return (self.get_shape_in() is not None) and (self.get_shape_out() is not None)

    def infer_shape(self):
        if self.is_configured():
            return
        shape_in = self.get_shape_in()
        if shape_in is not None:
            shape_out = self._infer(shape_in)
            self.set_shape_out(shape_out)
