from deepx.core import RecurrentLayer
from deepx.backend import T

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

    def _initial_state(self, data):
        batch_size = T.shape(data)[0]
        dim_out = self.get_dim_out()[0]
        return [T.zeros([batch_size, dim_out]), T.zeros([batch_size, dim_out])]

    def initialize(self):
        dim_in, dim_out = self.get_dim_in(), self.get_dim_out()
        self.create_lstm_parameters(dim_in[0], dim_out[0])

    def _step(self, X, state, params=None):
        previous_hidden, previous_state = state
        params = self.parameters
        Wi, Ui, bi = self.get_parameter_list('W_ix', 'U_ih', 'b_i', params=params)
        if self.use_input_peep:
            Pi = params['P_i']
            input_gate = T.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + T.dot(previous_state, Pi) + bi)
        else:
            input_gate = T.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + bi)

        Wg, Ug, bg = self.get_parameter_list('W_gx', 'U_gh', 'b_g', params=params)
        candidate_state = T.tanh(T.dot(X, Wg) + T.dot(previous_hidden, Ug) + bg)

        if self.use_forget_gate:
            Wf, Uf, bf = self.get_parameter_list('W_fx', 'U_fh', 'b_f', params=params)
            if self.use_forget_peep:
                Pf = params['P_f']
                forget_gate = T.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + T.dot(previous_state, Pf) + bf)
            else:
                forget_gate = T.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + bf)
            state = candidate_state * input_gate + previous_state * forget_gate
        else:
            state = candidate_state * input_gate + previous_state * 0

        Wo, Uo, bo = self.get_parameter_list('W_ox', 'U_oh', 'b_o', params=params)
        if self.use_output_peep:
            Po = params['P_o']
            output_gate = T.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + T.dot(previous_state, Po) + bo)
        else:
            output_gate = T.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + bo)
        if self.use_tanh_output:
            output = output_gate * T.tanh(state)
        else:
            output = output_gate * state
        return [output, state]
