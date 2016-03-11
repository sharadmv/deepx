from .. import backend as T

def Sequence(data, max_length=None):
    batch_size = data.batch_size
    var = T.make_sequence(data.get_data(), max_length)
    data = data.next(var, data.get_shape_out())
    data.sequence = True
    data.sequence_length = max_length
    data.batch_size = batch_size
    return data

