import numpy as np

from .. import T
from ..core import Shape
from ..layer import RecurrentLayer

class LSTM(RecurrentLayer):

    def __init__(self, shape_in=None, shape_out=None,
                 use_forget_gate=True,
                 use_input_peep=False,
                 use_output_peep=False,
                 use_forget_peep=False,
                 use_tanh_output=True, **kwargs):
        super(LSTM, self).__init__(shape_in=shape_in, shape_out=shape_out, **kwargs)

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

    def infer(self, shape_in):
        return shape_in.copy(dim=self.get_shapes_out()[0].get_dim())

    def create_lstm_parameters(self, shape_in, shape_out):
        self.create_parameter('W_ix', [shape_in, shape_out])
        self.create_parameter('U_ih', [shape_out, shape_out])
        self.create_parameter('b_i', [shape_out])

        self.create_parameter('W_ox', [shape_in, shape_out])
        self.create_parameter('U_oh', [shape_out, shape_out])
        self.create_parameter('b_o', [shape_out])

        if self.use_forget_gate:
            self.create_parameter('W_fx', [shape_in, shape_out])
            self.create_parameter('U_fh', [shape_out, shape_out])
            self.create_parameter('b_f', [shape_out])

        self.create_parameter('W_gx', [shape_in, shape_out])
        self.create_parameter('U_gh', [shape_out, shape_out])
        self.create_parameter('b_g', [shape_out])

        if self.use_input_peep:
            self.create_parameter('P_i', [shape_out, shape_out])
        if self.use_output_peep:
            self.create_parameter('P_o', [shape_out, shape_out])
        if self.use_forget_peep:
            self.create_parameter('P_f', [shape_out, shape_out])

    def create_initial_state(self, input_data, stateful, shape_index=1):
        batch_size = T.shape(input_data)[1]
        dim_out = self.get_dim_out()
        if stateful:
            if not isinstance(batch_size, int):
                raise TypeError("batch_size must be set for stateful RNN.")
            return [T.variable(np.zeros((batch_size, dim_out))), T.variable(np.zeros((batch_size, dim_out)))]
        return [T.alloc(0, (batch_size, dim_out), unbroadcast=shape_index), T.alloc(0, (batch_size, dim_out), unbroadcast=shape_index)]

    def initialize(self):
        dim_in, dim_out = self.get_dim_in(), self.get_dim_out()
        self.create_lstm_parameters(dim_in, dim_out)

    def step(self, X, states, **kwargs):
        previous_hidden, previous_state = states
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
        return output, [output, state]

class MaxoutLSTM(LSTM):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 4)
        super(MaxoutLSTM, self).__init__(*args, **kwargs)

    def create_lstm_parameters(self, shape_in, shape_out):
        self.create_parameter('W_ix', [shape_in, shape_out])
        self.create_parameter('U_ih', [shape_out, shape_out])
        self.create_parameter('b_i', [shape_out])

        self.create_parameter('W_ox', [shape_in, shape_out])
        self.create_parameter('U_oh', [shape_out, shape_out])
        self.create_parameter('b_o', [shape_out])

        if self.use_forget_gate:
            self.create_parameter('W_fx', [shape_in, shape_out])
            self.create_parameter('U_fh', [shape_out, shape_out])
            self.create_parameter('b_f', [shape_out])

        self.create_parameter('W_gx', [self.k, shape_in, shape_out])
        self.create_parameter('U_gh', [self.k, shape_out, shape_out])
        self.create_parameter('b_g', (self.k, [shape_out]))

        if self.use_input_peep:
            self.create_parameter('P_i', [shape_out, shape_out])
        if self.use_output_peep:
            self.create_parameter('P_o', [shape_out, shape_out])
        if self.use_forget_peep:
            self.create_parameter('P_f', [shape_out, shape_out])

    def step(self, X, states, **kwargs):
        previous_hidden, previous_state = states
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
        return output, [output, state]
