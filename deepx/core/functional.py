from abc import abstractmethod
from .unary import UnaryNode

__all__ = ["FunctionalNode"]

class FunctionalNode(UnaryNode):

    @abstractmethod
    def func(self, inputs):
        pass

    def get_outputs(self, *args, **kwargs):
        net = self.func(args)
        return net.get_outputs(**kwargs)

    @abstractmethod
    def get_shapes_out(self):
        pass

    @abstractmethod
    def get_num_outputs(self):
        pass
