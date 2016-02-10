from .. import backend as T
import numpy as np

from ..node import RecurrentNode, Data

class LSTM(RecurrentNode):

    def __init__(self, shape_in, shape_out=None,
                 use_forget_gate=True,
                 use_input_peep=False,
                 use_output_peep=False,
                 use_forget_peep=False,
                 use_tanh_output=True,
                 stateful=False):

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

        self.stateful = stateful
        self.states = None

    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, output_dim)
        N = T.shape(X)[1]
        return [T.alloc(0, (N, self.get_shape_out()), unbroadcast=1),
                T.alloc(0, (N, self.get_shape_out()), unbroadcast=1)]

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        batch_size = self.batch_size
        output_shape = self.get_shape_out()
        if not batch_size:
            raise Exception()
        if self.states is not None:
            for state in self.states:
                T.set_value(state,
                            np.zeros((batch_size, output_shape)))
        else:
            self.states = [T.zeros((batch_size, output_shape)),
                           T.zeros((batch_size, output_shape))]

    def copy(self):
        return LSTM(self.get_shape_in(),
                    self.get_shape_out(),
                    use_forget_gate=self.use_forget_gate,
                    use_input_peep=self.use_input_peep,
                    use_output_peep=self.use_output_peep,
                    use_forget_peep=self.use_forget_peep,
                    use_tanh_output=self.use_tanh_output,
                    stateful=self.stateful
                    )

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

    def initialize(self):
        shape_in, shape_out = self.get_shape_in(), self.get_shape_out()
        self.create_lstm_parameters(shape_in, shape_out)

    def _forward(self, X):
        S, N, D = T.shape(X)

        def step(input, previous):
            previous_hidden, previous_state = previous
            lstm_hidden, state = self.step(input, previous_hidden, previous_state)
            return lstm_hidden, [lstm_hidden, state]

        if self.stateful:
            if self.states is None:
                self.reset_states(X)
            hidden, state = self.states
        else:
            hidden, state = self.get_initial_states(X)

        last_output, output, new_state = T.rnn(step,
                              X,
                              [hidden, state])
        if self.stateful:
            for state, ns in zip(self.states, new_state):
                self.add_update(state, ns)
        return output

    def step(self, X, previous_hidden, previous_state):
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

    def get_previous_zeros(self, N):
        return T.alloc(0, (N, self.get_shape_out())), T.alloc(0, (N, self.get_shape_out()))
