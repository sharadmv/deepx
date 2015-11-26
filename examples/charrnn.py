import numpy as np
import theano
theano.config.on_unused_input = 'ignore'
import logging
logging.basicConfig(level=logging.DEBUG)
from argparse import ArgumentParser
from deepx.dataset import CharacterSequence, OneHotEncoding, WindowedBatcher, NumberSequence
from deepx.sequence import CharacterRNN
from deepx.optimize import RMSProp

from deepx.train import Trainer


def parse_args():
    argparser = ArgumentParser()

    argparser.add_argument("text_file")

    return argparser.parse_args()

def generate(length, temperature):
    results = charrnn.generate(
        np.eye(len(encoding))[encoding.encode("i")],
        length,
        temperature).argmax(axis=1)
    return NumberSequence(results).decode(encoding)

if __name__ == "__main__":

    args = parse_args()

    with open(args.text_file) as fp:
        text = fp.read()

    seq = CharacterSequence.from_string(text)

    encoding = OneHotEncoding()
    encoding.build_encoding([seq])

    batcher = WindowedBatcher([seq.encode(encoding)], [encoding], 10, 10)

    charrnn = CharacterRNN(
        'charrnn',
        len(encoding),
        len(encoding),
        100,
        2,
    )

    charrnn.compile_method("generate")

    rmsprop = RMSProp(charrnn)
    trainer = Trainer(rmsprop, batcher)

