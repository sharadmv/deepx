import theano
import theano.tensor as T
import numpy as np

from ..node import RecurrentNode, Data

class LSTM(RecurrentNode):

    def __init__(self, shape_in, shape_out=None,
                 use_forget_gate=True,
                 use_input_peep=False,
                 use_output_peep=False,
                 use_forget_peep=False,
                 use_tanh_output=True):

        super(LSTM, self).__init__()
        if shape_out is None:
            self.shape_in = None
            self.shape_out = shape_in
        else:
            self.shape_in = shape_in
            self.shape_out = shape_out

        self.use_forget_gate = use_forget_gate
        self.use_input_peep = use_input_peep
        self.use_output_peep = use_output_peep
        self.use_forget_peep = use_forget_peep
        self.use_tanh_output = use_tanh_output

    def _infer(self, shape_in):
        return self.shape_out

    def recurrent_forward(self, X, previous):
        out, state = self.forward(X, previous)
        return out, (out, state)

    def forward(self, X, previous):
        output, state = self._forward(X.get_data(), previous)
        return Data(output, self.get_shape_out()), Data(state, self.get_shape_out())


    def initialize(self):

        self.Wi = self.init_parameter('W_ix', (self.shape_in, self.shape_out))
        self.Ui = self.init_parameter('U_ih', (self.shape_out, self.shape_out))
        self.bi = self.init_parameter('b_i', self.shape_out)

        self.Wo = self.init_parameter('W_ox', (self.shape_in, self.shape_out))
        self.Uo = self.init_parameter('U_oh', (self.shape_out, self.shape_out))
        self.bo = self.init_parameter('b_o', self.shape_out)

        if self.use_forget_gate:
            self.Wf = self.init_parameter('W_fx', (self.shape_in, self.shape_out))
            self.Uf = self.init_parameter('U_fx', (self.shape_out, self.shape_out))
            self.bf = self.init_parameter('b_f', self.shape_out)

        self.Wg = self.init_parameter('W_gx', (self.shape_in, self.shape_out))
        self.Ug = self.init_parameter('U_gx', (self.shape_out, self.shape_out))
        self.bg = self.init_parameter('b_g', self.shape_out)

        if self.use_input_peep:
            self.Pi = self.init_parameter('P_i', (self.shape_out, self.shape_out))
        if self.use_output_peep:
            self.Po = self.init_parameter('P_o', (self.shape_out, self.shape_out))
        if self.use_forget_peep:
            self.Pf = self.init_parameter('P_f', (self.shape_out, self.shape_out))

    # def _forward(self, X):
        # S, N, D = X.shape

        # H = self.shape_out

        # def step(input, previous_hidden, previous_state):
            # lstm_hidden, state = self.step(input, previous_hidden, previous_state)
            # return lstm_hidden, state

        # hidden = T.alloc(np.array(0).astype(theano.config.floatX), N, H)
        # state = T.alloc(np.array(0).astype(theano.config.floatX), N, H)

        # (lstm_out, _), updates = theano.scan(step,
                              # sequences=[X],
                              # outputs_info=[
                                            # hidden,
                                            # state,
                                           # ],
                              # n_steps=S)
        # return lstm_out

    def _forward(self, X, previous):
        previous_hidden, previous_state = previous
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

    def get_previous_zeros(self, N):
        return (T.alloc(np.array(0).astype(theano.config.floatX), N, self.get_shape_out()),
                T.alloc(np.array(0).astype(theano.config.floatX), N, self.get_shape_out()))

