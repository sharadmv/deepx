import theano.tensor as T

from model import ParameterModel
from softmax import Softmax

def FeedForwardLayer(*args):
    class FeedForwardLayer(ParameterModel):

        def __init__(self, name, n_input, n_output,
                     activation='tanh'):
            super(FeedForwardLayer, self).__init__(name)
            self.n_input = n_input
            self.n_output = n_output

            self.init_parameter('W', self.initialize_weights((self.n_input, self.n_output)))
            self.init_parameter('b', self.initialize_weights((self.n_output,)))
            if activation == 'tanh':
                self.activation = self.tanh
            elif activation == 'relu':
                self.activation = self.relu
            elif activation == 'logistic':
                self.activation = self.logistic

        def forward(self, X):
            return self.activation(T.dot(X, self.get_parameter('W')) + self.get_parameter('b'))

        def logistic(self, X):
            return T.nnet.sigmoid(X)

        def tanh(self, X):
            return T.tanh(X)

        def relu(self, X):
            return T.nnet.relu(X)

        def state(self):
            state_params = {}
            for param, value in self.parameters.items():
                state_params[param] = value.get_value()
            return {
                'name': self.name,
                'n_input': self.n_input,
                'n_output': self.n_output,
                'parameters': state_params
            }

        def load(self, state):
            self.name = state['name']
            self.n_input = state['n_input']
            self.n_output = state['n_output']
            for param, value in state['parameters'].items():
                self.set_parameter_value(param, value)

    return FeedForwardLayer(*args)

def MultilayerPerceptron(*args, **kwargs):
    class MultilayerPerceptron(ParameterModel):

        def __init__(self, name, n_input, n_hidden, n_output, n_layers=2):
            super(MultilayerPerceptron, self).__init__(name)

            self.n_input = n_input
            self.n_hidden = n_hidden
            self.n_output = n_output
            self.n_layers = n_layers
            assert self.n_layers >= 1
            self.output_layer = Softmax('%s-output' % name,
                                        self.n_hidden,
                                        self.n_output)
            self.layers = [FeedForwardLayer('%s-input-layer' % name,
                                            self.n_input,
                                            self.n_hidden)]
            for i in xrange(self.n_layers - 1):
                self.layers.append(FeedForwardLayer('%s-layer-%u' % (name, i),
                                            self.n_hidden,
                                            self.n_hidden))

        def forward(self, X):
            out = X
            for layer in self.layers:
                out = layer.forward(out)
            out = self.output_layer.forward(out, 1.0)
            return out

        def get_parameters(self):
            params = []
            for layer in self.layers:
                params += layer.get_parameters()
            params.extend(self.output_layer.get_parameters())
            return params

        def state(self):
            return {
                'name': self.name,
                'n_input': self.n_input,
                'n_hidden': self.n_hidden,
                'n_output': self.n_output,
                'n_layers': self.n_layers,
                'output_layer': self.output_layer.state(),
                'layers': [layer.state() for layer in self.layers],
            }

        def load(self, state):
            self.name = state['name']
            self.n_input = state['n_input']
            self.n_hidden = state['n_hidden']
            self.n_output= state['n_output']
            self.n_layers = state['n_layers']
            self.output_layer.load(state['output_layer'])
            for layer, layer_state in zip(self.layers, state['layers']):
                layer.load(layer_state)

    return MultilayerPerceptron(*args, **kwargs)
