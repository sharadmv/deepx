import numpy as np

class Sequence(object):

    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def encode(self, encoding):
        return encoding.encode_sequence(self)

    def decode(self, encoding):
        return encoding.decode_sequence(self)

    def replicate(self, n_times):
        raise NotImplementedError

    def iter(self):
        return iter(self.seq)

    def __str__(self):
        return str(self.seq)

class SingletonSequence(Sequence):

    def __init__(self, obj):
        self.seq = [obj]

class NumberSequence(Sequence):

    def __init__(self, seq):
        self.seq = np.array(seq)
        if self.seq.ndim == 1:
            self.seq = self.seq[:, np.newaxis]

    def replicate(self, n_times):
        return NumberSequence(np.tile(self.seq, (n_times, 1)))

    def stack(self, num_seq):
        assert len(num_seq) == len(self), "Sequences must be same length"
        return NumberSequence(np.hstack([self.seq, num_seq.seq]))

    def __len__(self):
        return self.seq.shape[0]

class CharacterSequence(Sequence):

    def __init__(self, seq):
        self.seq = seq

    def concatenate(self, char_seq):
        return CharacterSequence(self.seq + char_seq.seq)

    @staticmethod
    def from_string(string, lower=False):
        string = string.replace("\x7f", "")
        if lower:
            return CharacterSequence(list(string.lower()))
        return CharacterSequence(list(string))

    def __str__(self):
        return ''.join(self.seq)
