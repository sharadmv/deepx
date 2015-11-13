import theano
import numpy as np
import theano.tensor as T

from theanify import theanify

from model import ParameterModel

from theano.tensor.shared_randomstreams import RandomStreams

def LSTMLayer(*args, **kwargs):
    class LSTMLayer(ParameterModel):

        def __init__(self, name, n_input, n_output,
                    use_forget_gate=True,
                    use_input_peep=False, use_output_peep=False, use_forget_peep=False,
                    use_tanh_output=True, seed=None, rng=None):
            super(LSTMLayer, self).__init__(name)

            self.rng = rng or RandomStreams(seed)

            self.n_input = n_input
            self.n_output = n_output

            self.use_forget_gate = use_forget_gate
            self.use_input_peep = use_input_peep
            self.use_output_peep = use_output_peep
            self.use_forget_peep = use_forget_peep
            self.use_tanh_output = use_tanh_output

            np.random.seed(seed)

            self.init_parameter('W_ix', self.initialize_weights((self.n_input, self.n_output)))
            self.init_parameter('U_ih', self.initialize_weights((self.n_output, self.n_output)))
            self.init_parameter('b_i', self.initialize_weights((self.n_output,)))

            self.init_parameter('W_ox', self.initialize_weights((self.n_input, self.n_output)))
            self.init_parameter('U_oh', self.initialize_weights((self.n_output, self.n_output)))
            self.init_parameter('b_o', self.initialize_weights((self.n_output,)))

            self.init_parameter('W_fx', self.initialize_weights((self.n_input, self.n_output)))
            self.init_parameter('U_fh', self.initialize_weights((self.n_output, self.n_output)))
            self.init_parameter('b_f', self.initialize_weights((self.n_output,)))

            self.init_parameter('W_gx', self.initialize_weights((self.n_input, self.n_output)))
            self.init_parameter('U_gh', self.initialize_weights((self.n_output, self.n_output)))
            self.init_parameter('b_g', self.initialize_weights((self.n_output,)))

            if self.use_input_peep:
                self.init_parameter('P_i', self.initialize_weights((self.n_output, self.n_output)))
            if self.use_output_peep:
                self.init_parameter('P_o', self.initialize_weights((self.n_output, self.n_output)))
            if self.use_forget_peep:
                self.init_parameter('P_f', self.initialize_weights((self.n_output, self.n_output)))

        @theanify(T.matrix('X'), T.matrix('previous_hidden'), T.matrix('previous_state'),
                  T.fscalar('dropout_prob'))
        def step(self, X, previous_hidden, previous_state, dropout_prob):
            """
            Parameters:
                X               - B x D (B is batch size, D is dimension)
                previous_hidden - B x O (B is batch size, O is output size)
                previous_state  - B x O (B is batch size, O is output size)
            Returns:
                output          - B x O
                state           - B x O
            """
            Wi = self.get_parameter('W_ix')
            Wo = self.get_parameter('W_ox')
            if self.use_forget_gate:
                Wf = self.get_parameter('W_fx')
            Wg = self.get_parameter('W_gx')

            Ui = self.get_parameter('U_ih')
            Uo = self.get_parameter('U_oh')
            Uf = self.get_parameter('U_fh')
            Ug = self.get_parameter('U_gh')

            bi = self.get_parameter('b_i')
            bo = self.get_parameter('b_o')
            bf = self.get_parameter('b_f')
            bg = self.get_parameter('b_g')

            if self.use_input_peep:
                Pi = self.get_parameter('P_i')
                input_gate = T.nnet.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + T.dot(previous_state, Pi) + bi)
            else:
                input_gate = T.nnet.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + bi)
            candidate_state = T.tanh(T.dot(X, Wg) + T.dot(previous_hidden, Ug) + bg)

            if self.use_forget_gate:
                if self.use_forget_peep:
                    Pf = self.get_parameter('P_f')
                    forget_gate = T.nnet.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + T.dot(previous_state, Pf) + bf)
                else:
                    forget_gate = T.nnet.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + bf)
                state = candidate_state * input_gate + previous_state * forget_gate
            else:
                state = candidate_state * input_gate + previous_state * 0

            if self.use_output_peep:
                Po = self.get_parameter('P_o')
                output_gate = T.nnet.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + T.dot(previous_state, Po) + bo)
            else:
                output_gate = T.nnet.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + bo)
            if self.use_tanh_output:
                output = output_gate * T.tanh(state)
            else:
                output = output_gate * state
            return output, state

        def state(self):
            state_params = {}
            for param, value in self.parameters.items():
                state_params[param] = value.get_value()
            return {
                'name': self.name,
                'n_input': self.n_input,
                'n_output': self.n_output,
                'use_forget_gate': self.use_forget_gate,
                'use_input_peep' : self.use_input_peep,
                'use_output_peep' : self.use_output_peep,
                'use_forget_peep' : self.use_forget_peep,
                'use_tanh_output' : self.use_tanh_output,
                'parameters': state_params
            }

        def load(self, state):
            self.name = state['name']
            self.n_input = state['n_input']
            self.n_output = state['n_output']
            self.use_forget_gate = state['use_forget_gate']
            self.use_input_peep = state['use_input_peep']
            self.use_output_peep = state['use_output_peep']
            self.use_forget_peep = state['use_forget_peep']
            self.use_tanh_output = state['use_tanh_output']
            for param, value in state['parameters'].items():
                self.set_parameter_value(param, value)

    return LSTMLayer(*args, **kwargs)

def LSTM(*args, **kwargs):
    class LSTM(ParameterModel):
        def __init__(self, name, n_input, n_hidden=10, n_layers=2, dropout_probability=0.0,
                     rng=None):
            super(LSTM, self).__init__(name)

            self.n_input = n_input
            self.n_hidden = n_hidden
            self.n_layers = n_layers
            assert self.n_layers >= 1
            self.layers = []
            self.dropout_probability = dropout_probability
            self.input_layer = LSTMLayer('%s-input' % name,
                                        self.n_input,
                                        self.n_hidden,
                                        rng=rng)
            for i in xrange(self.n_layers - 1):
                self.layers.append(LSTMLayer('%s-layer-%u' % (name, i),
                                            self.n_hidden,
                                            self.n_hidden,
                                            rng=rng))

        def forward(self, X, previous_state, previous_hidden):
            output, state = self.input_layer.step(X, previous_state[:, 0, :], previous_hidden[:, 0, :],
                                                  self.dropout_probability)
            hiddens, states = [output], [state]
            for i, layer in enumerate(self.layers):
                output, state = layer.step(output, previous_state[:, i + 1, :], previous_hidden[:, i + 1, :],
                                           self.dropout_probability)
                hiddens.append(output)
                states.append(state)
            return T.swapaxes(T.stack(*hiddens), 0, 1), T.swapaxes(T.stack(*states), 0, 1)

        def get_parameters(self):
            params = self.input_layer.get_parameters()
            for layer in self.layers:
                params += layer.get_parameters()
            return params

        def state(self):
            return {
                'name': self.name,
                'n_input': self.n_input,
                'n_hidden': self.n_hidden,
                'n_layers': self.n_layers,
                'input_layer': self.input_layer.state(),
                'layers': [layer.state() for layer in self.layers],
            }

        def load(self, state):
            self.name = state['name']
            self.n_input = state['n_input']
            self.n_hidden = state['n_hidden']
            self.n_layers = state['n_layers']
            self.input_layer.load(state['input_layer'])
            for layer, layer_state in zip(self.layers, state['layers']):
                layer.load(layer_state)

    return LSTM(*args, **kwargs)
