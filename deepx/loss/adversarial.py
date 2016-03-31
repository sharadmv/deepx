from .. import backend as T
from .loss import Loss

class AdversarialLoss(Loss):

    def __init__(self, loss, alpha=0.5, e=0.01):
        self.inner_loss = loss
        self.alpha = 0.5
        self.e = e
        self._grads = None

    def compute_loss(self, y):
        loss = self.inner_loss.compute_loss(y)
        dx = T.gradients(loss, self.get_inputs())[0]
        norm = T.expand_dims(T.norm(dx, 2, axis=1))
        dx /= norm
        ad_loss = self.inner_loss.compute_loss(y, transform=lambda x: x + dx)
        return self.alpha * loss + (1 - self.alpha) * ad_loss

    def get_activation(self, **kwargs):
        return self.inner_loss.get_activation(**kwargs)

    def get_updates(self):
        return self.inner_loss.get_updates()

    def get_inputs(self):
        return self.inner_loss.get_inputs()

    def get_parameters(self):
        return self.inner_loss.get_parameters()

