import numpy as np
from sequence import NumberSequence

class Encoding(object):

    START_TOKEN = '<STR>'
    STOP_TOKEN = '<EOS>'

    def __init__(self):
        self.include_start_token = False
        self.include_stop_token = False

    in_sequence_type = None
    out_sequence_type = None

    def build_encoding(self, sequences):
        raise NotImplementedError

    def encode(self, obj):
        raise NotImplementedError

    def decode(self, obj):
        raise NotImplementedError

    def encode_sequence(self, sequence):
        seq = [self.encode(s) for s in sequence.iter()]
        if self.include_start_token:
            seq = [self.encode(self.START_TOKEN)] + seq
        if self.include_stop_token:
            seq = seq + [self.encode(self.STOP_TOKEN)]
        return self.out_sequence_type(
            seq
        )

    def decode_sequence(self, sequence):
        return self.in_sequence_type(
            [self.decode(s) for s in sequence.iter()]
        )

    def __len__(self):
        return self.index

class IdentityEncoding(Encoding):

    def __init__(self, size):
        self.index = size

    def convert_representation(self, rep):
        return rep

class OneHotEncoding(Encoding):

    out_sequence_type = NumberSequence

    def __init__(self, include_start_token=True, include_stop_token=True):
        super(Encoding, self).__init__()
        self.include_start_token = include_start_token
        self.include_stop_token = include_stop_token
        self.forward_mapping = {}
        self.backward_mapping = []
        self.index = 0
        self.in_sequence_type = None

        if self.include_start_token:
            self.forward_mapping[self.START_TOKEN] = self.index
            self.backward_mapping.append(self.START_TOKEN)
            self.index += 1
        if self.include_stop_token:
            self.forward_mapping[self.STOP_TOKEN] = self.index
            self.backward_mapping.append(self.STOP_TOKEN)
            self.index += 1

    def build_encoding(self, sequences):
        for sequence in sequences:
            self.in_sequence_type = sequence.__class__
            for word in sequence.iter():
                if word not in self.forward_mapping:
                    self.forward_mapping[word] = self.index
                    self.backward_mapping.append(word)
                    self.index += 1

    def encode(self, word):
        if word not in self.forward_mapping:
            assert word in self.forward_mapping, "'%s' not in one-hot mapping" % word
        return self.forward_mapping[word]

    def decode(self, index):
        return self.backward_mapping[index]

    def convert_representation(self, rep):
        vec = np.zeros(self.index)
        vec[rep[0]] = 1
        return vec
