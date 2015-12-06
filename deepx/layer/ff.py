import theano.tensor as T

from layer import Layer

class Linear(Layer):
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__(n_in, n_out)

        self.W = self.init_parameter('W', (n_in, n_out))
        self.b = self.init_parameter('b', n_out)

    def activation(self, X):
        return X

    def get_layer_var(self):
        return T.matrix()

    def get_in_var(self):
        return T.matrix()

    def _forward(self, X):
        return self.activation(T.dot(X, self.W) + self.b)

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
