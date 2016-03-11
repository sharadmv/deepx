from .. import backend as T
import numpy as np

from ..node import RecurrentNode, Data

class LSTM(RecurrentNode):

    def __init__(self, *args, **kwargs):

        self.use_forget_gate = kwargs.get('use_forget_gate', True)
        self.use_input_peep = kwargs.get('use_input_peep', False)
        self.use_output_peep = kwargs.get('use_output_peep', False)
        self.use_forget_peep = kwargs.get('use_forget_peep', False)
        self.use_tanh_output = kwargs.get('use_tanh_output', True)

        super(LSTM, self).__init__(*args, **kwargs)


    def _infer(self, shape_in):
        return self.shape_out

    def create_lstm_parameters(self, shape_in, shape_out):
        self.init_parameter('W_ix', (shape_in, shape_out))
        self.init_parameter('U_ih', (shape_out, shape_out))
        self.init_parameter('b_i', shape_out)

        self.init_parameter('W_ox', (shape_in, shape_out))
        self.init_parameter('U_oh', (shape_out, shape_out))
        self.init_parameter('b_o', shape_out)

        if self.use_forget_gate:
            self.init_parameter('W_fx', (shape_in, shape_out))
            self.init_parameter('U_fh', (shape_out, shape_out))
            self.init_parameter('b_f', shape_out)

        self.init_parameter('W_gx', (shape_in, shape_out))
        self.init_parameter('U_gh', (shape_out, shape_out))
        self.init_parameter('b_g', shape_out)

        if self.use_input_peep:
            self.init_parameter('P_i', (shape_out, shape_out))
        if self.use_output_peep:
            self.init_parameter('P_o', (shape_out, shape_out))
        if self.use_forget_peep:
            self.init_parameter('P_f', (shape_out, shape_out))

    def get_initial_states(self, X, shape_index=1):
        if self.stateful:
            N = self.get_batch_size()
        else:
            N = T.shape(X)[shape_index]
        if self.stateful:
            if not N:
                raise Exception('Must set batch size for input')
            else:
                return [T.zeros((N, self.get_shape_out())),
                        T.zeros((N, self.get_shape_out()))]
        return [T.alloc(0, (N, self.get_shape_out()), unbroadcast=shape_index),
                T.alloc(0, (N, self.get_shape_out()), unbroadcast=shape_index)]

    def initialize(self):
        shape_in, shape_out = self.get_shape_in(), self.get_shape_out()
        self.create_lstm_parameters(shape_in, shape_out)

    def step(self, X, state):
        out, state = self._step(X.get_data(), state)
        return X.next(out, self.get_shape_out()), state

    def _step(self, X, state):
        previous_hidden, previous_state = state
        lstm_hidden, state = self.lstm_step(X, previous_hidden, previous_state)
        return lstm_hidden, [lstm_hidden, state]

    def _forward(self, X):
        S, N, D = T.shape(X)

        if self.stateful:
            if self.states is None:
                self.states = self.get_initial_states(None)
            hidden, state = self.states
        else:
            hidden, state = self.get_initial_states(X)

        _, output, new_state = T.rnn(self._step,
                              X,
                              [hidden, state])
        if self.stateful:
            for state, ns in zip(self.states, new_state):
                self.add_update(state, ns)
        return output

    def lstm_step(self, X, previous_hidden, previous_state):
        params = self.parameters
        Wi, Ui, bi = params['W_ix'], params['U_ih'], params['b_i']
        if self.use_input_peep:
            Pi = params['P_i']
            input_gate = T.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + T.dot(previous_state, Pi) + bi)
        else:
            input_gate = T.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + bi)

        Wg, Ug, bg = params['W_gx'], params['U_gh'], params['b_g']
        candidate_state = T.tanh(T.dot(X, Wg) + T.dot(previous_hidden, Ug) + bg)

        if self.use_forget_gate:
            Wf, Uf, bf = params['W_fx'], params['U_fh'], params['b_f']
            if self.use_forget_peep:
                Pf = params['P_f']
                forget_gate = T.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + T.dot(previous_state, Pf) + bf)
            else:
                forget_gate = T.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + bf)
            state = candidate_state * input_gate + previous_state * forget_gate
        else:
            state = candidate_state * input_gate + previous_state * 0

        Wo, Uo, bo = params['W_ox'], params['U_oh'], params['b_o']
        if self.use_output_peep:
            Po = params['P_o']
            output_gate = T.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + T.dot(previous_state, Po) + bo)
        else:
            output_gate = T.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + bo)
        if self.use_tanh_output:
            output = output_gate * T.tanh(state)
        else:
            output = output_gate * state
        return output, state
