from deepx.nn import Softmax, Tanh, predict, Sequence
from deepx.rnn import LSTM
from deepx.optimize import rmsprop, cross_entropy

if __name__ == "__main__":
    lstm = Sequence('X', 100) >> LSTM(100) >> LSTM(100) >> Softmax(10) | (predict, cross_entropy, rmsprop)
