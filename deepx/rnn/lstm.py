import numpy as np

from .. import backend as T
from ..core import RecurrentLayer, Shape

class LSTM(RecurrentLayer):

    def __init__(self, shape_in=None, shape_out=None,
                 use_forget_gate=True,
                 use_input_peep=False,
                 use_output_peep=False,
                 use_forget_peep=False,
                 use_tanh_output=True, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        if shape_out is not None:
            shape_in, shape_out = (shape_in, shape_out)
        elif shape_in is not None and shape_out is None:
            shape_in, shape_out = None, shape_in
        if shape_in is not None:
            self.set_shape_in([Shape(shape_in, sequence=True)])
        if shape_out is not None:
            self.set_shape_out([Shape(shape_out, sequence=True)])

        self.use_forget_gate = use_forget_gate
        self.use_input_peep = use_input_peep
        self.use_output_peep = use_output_peep
        self.use_forget_peep = use_forget_peep
        self.use_tanh_output = use_tanh_output

        if shape_out is not None:
            shape_in, shape_out = (shape_in, shape_out)
        elif shape_in is not None and shape_out is None:
            shape_in, shape_out = None, shape_in
        else:
            raise Exception("Need to specify LSTM shape")

    def _infer(self, shape_in):
        return shape_in.copy(dim=self.get_dim_out())

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

    def get_initial_states(self, input_data=None, shape_index=1):
        hidden = super(LSTM, self).get_initial_states(input_data=input_data, shape_index=shape_index)
        state = super(LSTM, self).get_initial_states(input_data=input_data, shape_index=shape_index)
        return hidden + state

    def initialize(self):
        dim_in, dim_out = self.get_dim_in(), self.get_dim_out()
        self.create_lstm_parameters(dim_in, dim_out)

    def _step(self, X, state):
        previous_hidden, previous_state = state
        lstm_hidden, state = self.lstm_step(X, previous_hidden, previous_state)
        return lstm_hidden, [lstm_hidden, state]

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

class MaxoutLSTM(LSTM):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 4)
        super(MaxoutLSTM, self).__init__(*args, **kwargs)

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

        self.init_parameter('W_gx', (self.k, shape_in, shape_out))
        self.init_parameter('U_gh', (self.k, shape_out, shape_out))
        self.init_parameter('b_g', (self.k, shape_out))

        if self.use_input_peep:
            self.init_parameter('P_i', (shape_out, shape_out))
        if self.use_output_peep:
            self.init_parameter('P_o', (shape_out, shape_out))
        if self.use_forget_peep:
            self.init_parameter('P_f', (shape_out, shape_out))


    def lstm_step(self, X, previous_hidden, previous_state):
        params = self.parameters
        Wi, Ui, bi = params['W_ix'], params['U_ih'], params['b_i']
        if self.use_input_peep:
            Pi = params['P_i']
            input_gate = T.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + T.dot(previous_state, Pi) + bi)
        else:
            input_gate = T.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + bi)

        Wg, Ug, bg = params['W_gx'], params['U_gh'], params['b_g']
        candidate_state = T.max(T.dot(X, Wg) + T.dot(previous_hidden, Ug) + bg, axis=1)

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
