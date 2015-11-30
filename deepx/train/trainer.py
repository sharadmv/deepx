import logging

class Trainer(object):

    def __init__(self, optimizer, batcher, learning_curve=None,
                 callbacks=None):
        self.optimizer = optimizer
        self.batcher = batcher
        self.reset()
        self.learning_curve = learning_curve
        self.callbacks = callbacks or []

    def train(self, n_iterations, *args):
        loss = []
        for i in xrange(n_iterations):

            if self.batcher.batch_index == 0:
                self.state = self.optimizer.get_initial_state(self.batcher.batch_size)
            X, y = self.batcher.next_batch()
            self.current_loss, state = self.optimizer.train(X, self.state, y, *args)
            self.state = state[-1]
            self.losses.append(self.current_loss)
            loss.append(self.current_loss)
            if self.running_loss is None:
                self.running_loss = self.current_loss
            else:
                self.running_loss = 0.95 * self.running_loss + 0.05 * self.current_loss
            self.running_losses.append(self.running_loss)
            logging.info("Iteration %u (%u): %f [%f]" % (
                i,
                self.index,
                self.current_loss,
                self.running_loss
            ))
            self.index = self.batcher.batch_index
            self.total_iterations += 1
            if self.total_iterations % 10 == 0:
                for callback in self.callbacks:
                    callback()
        return loss

    def reset(self):
        self.batcher.batch_index = 0
        self.running_loss = None
        self.current_loss = None
        self.state = self.optimizer.get_initial_state(self.batcher.batch_size)
        self.index = self.batcher.batch_index
        self.running_losses = []
        self.losses = []
        self.total_iterations = 0
