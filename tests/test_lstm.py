from base import BaseTest

import numpy as np
import deepx.backend as T
from deepx.nn import Vector
from deepx.rnn import LSTM, Sequence

import unittest

np.random.seed(1)

def logistic(x):
    return 1.0/(1.0 + np.exp(-x))

class LSTMBase(BaseTest):
    def zero_lstm(self, lstm):
        for param in lstm.parameters:
            lstm.set_parameter_value(param, lstm.get_parameter_value(param) * 0)

    def set_weights(self, lstm, val):
        self.zero_lstm(lstm)
        for param in lstm.parameters:
            if param[0] != "b":
                lstm.set_parameter_value(param, lstm.get_parameter_value(param) + val)

    def make_lstm_step(self, lstm):
        X = T.placeholder(ndim=2)
        H = T.placeholder(ndim=2)
        S = T.placeholder(ndim=2)
        out, state = lstm.right.lstm_step(X, H, S)
        return T.function([X, H, S],[out, state])

    def lstm_forward(self, X, hidden, state, weights):
        input_gate = logistic(np.dot(X, weights) + np.dot(hidden, weights))
        output_gate = logistic(np.dot(X, weights) + np.dot(hidden, weights))
        forget_gate = logistic(np.dot(X, weights) + np.dot(hidden, weights))

        candidate_state = np.tanh(np.dot(X, weights) + np.dot(hidden, weights))

        state = candidate_state  * input_gate + state * forget_gate
        out = output_gate * np.tanh(state)
        return out, state

class TestSimpleLSTM(LSTMBase):

    def setUp(self):
        self.lstm = Sequence(Vector(1)) >> LSTM(1, 1, use_forget_gate=False)
        self.lstm_forget = Sequence(Vector(1)) >> LSTM(1, 1, use_forget_gate=True)

    def test_input_gate(self):
        self.set_weights(self.lstm.right, 1)
        X = np.ones((1, 1))
        S = np.zeros((1, 1))
        H = np.zeros((1, 1))
        step = self.make_lstm_step(self.lstm)

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = step(X, H, S)

        np.testing.assert_almost_equal(lstm_out, out)
        np.testing.assert_almost_equal(lstm_state, state)

        np.testing.assert_almost_equal(lstm_out, out)

        Wi = self.lstm.right.get_parameter_value('W_ix')
        self.lstm.right.set_parameter_value('W_ix', Wi * 0)

        input_gate = 0.5
        state = np.tanh(1) * input_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = step(X, H, S)
        np.testing.assert_almost_equal(lstm_out, out)
        np.testing.assert_almost_equal(lstm_state, state)

    def test_output_gate(self):
        self.set_weights(self.lstm.right, 1)
        X = np.ones((1, 1))
        S = np.zeros((1, 1))
        H = np.zeros((1, 1))
        step = self.make_lstm_step(self.lstm)

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = step(X, H, S)

        np.testing.assert_almost_equal(lstm_out, out)
        np.testing.assert_almost_equal(lstm_state, state)

        Wo = self.lstm.right.get_parameter_value('W_ox')
        self.lstm.right.set_parameter_value('W_ox', Wo * 0)

        input_gate = logistic(X)
        state = np.tanh(1) * input_gate
        output_gate = logistic(0)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = step(X, H, S)
        np.testing.assert_almost_equal(lstm_out, out)
        np.testing.assert_almost_equal(lstm_state, state)

    def test_forget_gate(self):
        self.set_weights(self.lstm_forget.right, 1)
        X = np.ones((1, 1))
        S = np.ones((1, 1))
        H = np.zeros((1, 1))
        step = self.make_lstm_step(self.lstm_forget)

        input_gate = logistic(X)
        forget_gate = logistic(X)
        state = np.tanh(1) * input_gate + forget_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = step(X, H, S)

        np.testing.assert_almost_equal(lstm_out, out)
        np.testing.assert_almost_equal(lstm_state, state)

        Wf = self.lstm_forget.right.get_parameter_value('W_fx')
        self.lstm_forget.right.set_parameter_value('W_fx', Wf * 0)

        input_gate = logistic(X)
        forget_gate = 0.5
        state = np.tanh(1) * input_gate + forget_gate
        output_gate = logistic(1)
        out = output_gate * np.tanh(state)
        lstm_out, lstm_state = step(X, H, S)

        np.testing.assert_almost_equal(lstm_out, out)
        np.testing.assert_almost_equal(lstm_state, state)

class TestStatefulLSTM(LSTMBase):

    def setUp(self):
        self.lstm = Sequence(Vector(1, 1), 1) >> LSTM(1, 1, stateful=True)
        self.lstm = Sequence(Vector(1, 1), 1) >> LSTM(1, 1, stateful=True)

    def test_stateful_lstm(self):
        self.lstm.reset_states()
        self.set_weights(self.lstm.right, 2)

        X = np.random.normal(size=(1, 1, 1))
        weights = np.ones((self.lstm.get_shape_in(), self.lstm.get_shape_out())) * 2
        state = np.zeros((1, self.lstm.get_shape_out()))
        out = np.zeros((1, self.lstm.get_shape_out()))

        for i in range(1000):
            out, state = self.lstm_forward(X, out, state, weights)

            lstm_out = self.lstm.predict(X)
            lstm_state = T.get_value(self.lstm.right.states[1])

            np.testing.assert_almost_equal(lstm_out[0], out[0], 4)
            np.testing.assert_almost_equal(lstm_state, state[0], 4)


    def test_stateful_lstm2(self):
        weights = np.ones((self.lstm.get_shape_in(), self.lstm.get_shape_out()))
        for _ in range(10):
            X = np.random.normal(size=(10, 1, 1))

            for s in range(1, 3):
                state = np.zeros((1, self.lstm.get_shape_out()))
                out = np.zeros((1, self.lstm.get_shape_out()))
                for i in range(s):
                    out, state = self.lstm_forward(X[i], out, state, weights)

                lstm = Sequence(Vector(1, 1), s) >> LSTM(1, 1, stateful=True)
                self.set_weights(lstm.right, 1)
                lstm_out = lstm.predict(X[:s])[-1]
                lstm_hidden = T.get_value(lstm.right.states[0])
                lstm_state = T.get_value(lstm.right.states[1])


                np.testing.assert_almost_equal(lstm_hidden, lstm_out, 5)
                np.testing.assert_almost_equal(lstm_out, out, 5)
                np.testing.assert_almost_equal(out, lstm_hidden, 5)
                np.testing.assert_almost_equal(lstm_state, state, 5)

if __name__ == "__main__":
    unittest.main()
