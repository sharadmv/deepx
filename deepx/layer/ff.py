import theano.tensor as T

from layer import FeedForwardLayer

class Linear(FeedForwardLayer):

    def __init__(self, n_in, n_out):

        super(Linear, self).__init__(n_in, n_out)

        self.W = self.init_parameter('W', (n_in, n_out))
        self.b = self.init_parameter('b', n_out)

    def activation(self, X):
        return X

    def _forward(self, X):
        return self.activation(T.dot(X, self.W) + self.b)

    def __str__(self):
        return "%s(%u, %u)" % (self.__class__.__name__,
                               self.n_in, self.n_out)

class Softmax(Linear):

    def activation(self, X):
        return T.nnet.softmax(X)

class Sigmoid(Linear):

    def activation(self, X):
        return T.nnet.sigmoid(X)

class Tanh(Linear):

    def activation(self, X):
        return T.tanh(X)

class Relu(Linear):

    def activation(self, X):
        return T.nnet.relu(X)
