from deepx.backend import T
from deepx.nn import Softmax, Repeat
from deepx.rnn import LSTM

from jax import jit

net = Repeat(LSTM(50), 3)  >> Softmax(2)
result = net(T.ones([4, 50, 2]))
