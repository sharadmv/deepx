import theano.tensor as T

from rnn import StatefulRecurrentLayer

class LSTM(StatefulRecurrentLayer):

    def __init__(self, n_in, n_out,
                 use_forget_gate=True,
                 use_input_peep=False,
                 use_output_peep=False,
                 use_forget_peep=False,
                 use_tanh_output=True):

        super(LSTM, self).__init__(n_in, n_out)

        self.use_forget_gate = use_forget_gate
        self.use_input_peep = use_input_peep
        self.use_output_peep = use_output_peep
        self.use_forget_peep = use_forget_peep
        self.use_tanh_output = use_tanh_output

        self.Wi = self.init_parameter('W_ix', (self.n_in, self.n_out))
        self.Ui = self.init_parameter('U_ih', (self.n_out, self.n_out))
        self.bi = self.init_parameter('b_i', self.n_out)

        self.Wo = self.init_parameter('W_ox', (self.n_in, self.n_out))
        self.Uo = self.init_parameter('U_oh', (self.n_out, self.n_out))
        self.bo = self.init_parameter('b_o', self.n_out)

        if self.use_forget_gate:
            self.Wf = self.init_parameter('W_fx', (self.n_in, self.n_out))
            self.Uf = self.init_parameter('U_fx', (self.n_out, self.n_out))
            self.bf = self.init_parameter('b_f', self.n_out)

        self.Wg = self.init_parameter('W_gx', (self.n_in, self.n_out))
        self.Ug = self.init_parameter('U_gx', (self.n_out, self.n_out))
        self.bg = self.init_parameter('b_g', self.n_out)

        if self.use_input_peep:
            self.Pi = self.init_parameter('P_i', (self.n_out, self.n_out))
        if self.use_output_peep:
            self.Po = self.init_parameter('P_o', (self.n_out, self.n_out))
        if self.use_forget_peep:
            self.Pf = self.init_parameter('P_f', (self.n_out, self.n_out))

    def _forward(self, X, previous_hidden):
        previous_hidden, previous_state = previous_hidden
        if self.use_input_peep:
            input_gate = T.nnet.sigmoid(T.dot(X, self.Wi) + T.dot(previous_hidden, self.Ui) + T.dot(previous_state, self.Pi) + self.bi)
        else:
            input_gate = T.nnet.sigmoid(T.dot(X, self.Wi) + T.dot(previous_hidden, self.Ui) + self.bi)
        candidate_state = T.tanh(T.dot(X, self.Wg) + T.dot(previous_hidden, self.Ug) + self.bg)

        if self.use_forget_gate:
            if self.use_forget_peep:
                forget_gate = T.nnet.sigmoid(T.dot(X, self.Wf) + T.dot(previous_hidden, self.Uf) + T.dot(previous_state, self.Pf) + self.bf)
            else:
                forget_gate = T.nnet.sigmoid(T.dot(X, self.Wf) + T.dot(previous_hidden, self.Uf) + self.bf)
            state = candidate_state * input_gate + previous_state * forget_gate
        else:
            state = candidate_state * input_gate + previous_state * 0

        if self.use_output_peep:
            output_gate = T.nnet.sigmoid(T.dot(X, self.Wo) + T.dot(previous_hidden, self.Uo) + T.dot(previous_state, self.Po) + self.bo)
        else:
            output_gate = T.nnet.sigmoid(T.dot(X, self.Wo) + T.dot(previous_hidden, self.Uo) + self.bo)
        if self.use_tanh_output:
            output = output_gate * T.tanh(state)
        else:
            output = output_gate * state
        return output, state
