from deepx.nn import Softmax, Tanh, predict, Sequence, Vector
from deepx.rnn import LSTM
from deepx.optimize import rmsprop, cross_entropy

if __name__ == "__main__":
    lstm = Sequence(Vector('X', 100)) >> LSTM(100)
