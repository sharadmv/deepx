
class DimensionException(Exception):

    def __init__(self, in_layer, out_layer):
        message = "Error when joining layers: %s has expected output %u and %s has expected input %u." % (
            str(in_layer), in_layer.n_out,
            str(out_layer), out_layer.n_in
        )
        super(DimensionException, self).__init__(message)
