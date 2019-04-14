from deepx.backend import T
from deepx.nn import Softmax, Repeat
from deepx.rnn import LSTM

from jax import jit

net = LSTM(4, 50) >> Repeat(LSTM(50), 4) >> Softmax(2)
fast_net = jit(net)
