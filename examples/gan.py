import numpy as np

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    gen = Generate(Vector(10, 10, name='X') >> Repeat(LSTM(20), 2) >> Softmax(10), 10)

    disc = Repeat(LSTM(20), 2) >> Softmax(2)
    panel = [gen >> (disc.copy()) for _ in xrange(2)]
    losses = [CrossEntropy(l) for l in panel]

    final_loss = sum(losses) / 5.0

    rmsprop = RMSProp(final_loss)

    # discriminator = Sequence(Vector(100)) >> (Repeat(LSTM(1024), 2) >> Softmax(2))
    # panel = [Freeze(discriminator.copy()) for _ in xrange(5)]

    # generator = Generate(Vector(100) >> Repeat(LSTM(1024), 2) >> Softmax(100), 100)

    # gennet = Generate(Vector(100) >> Repeat(LSTM(1024), 2) >> Softmax(100), 100)
    # generator = generator.tie(gennet.node)

    # assert gennet.get_parameters() == generator.get_parameters()


    # generator_loss = sum([CrossEntropy(generator >> panel_member) for panel_member in panel]) / float(len(panel))

    # # Optimization for the generator (G)
    # rmsprop_G = RMSProp(generator_loss, clip_gradients=100)

    # # Optimization for the discrimator (D)
    # # gan.right.frozen = False
    # # assert len(discriminator.get_parameters()) > 0
    # rmsprop_D = RMSProp(CrossEntropy(discriminator), clip_gradients=100)
