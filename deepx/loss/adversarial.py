from .. import T
from ..core import Data
from .loss import Loss

class AdversarialLoss(Loss):

    def __init__(self, node, X, alpha=0.5, e=0.01):
        self.node = node
        self.X = X

        self.alpha = alpha
        self.e = e
        self._grads = None
        super(AdversarialLoss, self).__init__()

    def is_input(self):
        return self.node.is_input()

    def forward(self, *args, **kwargs):
        node_output = self.node.forward(*args, **kwargs)[0]
        X = self.X.get_placeholder()
        grads = T.gradients(node_output.get_placeholder(), [X])[0]
        X = X + self.e * T.sign(grads)
        data = Data.from_placeholder(X, self.X.dim, self.X.batch_size)
        inputs = [x if self.X != x else data for x in args]
        adversarial_output = self.node.forward(*inputs, **kwargs)[0]
        loss = self.alpha * node_output.get_placeholder() + (1 - self.alpha) * adversarial_output.get_placeholder()
        return [Data.from_placeholder(
            loss,
            node_output.dim,
            node_output.batch_size
        )]

    def get_shape_in(self):
        return self.node.get_shape_in()

    def get_shape_out(self):
        return ()

    def __str__(self):
        return self.__class__.__name__

    def get_activation(self, **kwargs):
        return self.inner_loss.get_activation(**kwargs)

    def get_updates(self):
        return self.node.get_updates()

    def get_network_inputs(self):
        return self.node.get_network_inputs()

    def get_inputs(self):
        return self.node.get_inputs()

    def get_parameters(self):
        return self.node.get_parameters()
