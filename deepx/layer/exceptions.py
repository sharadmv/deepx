
class DimensionException(Exception):

    def __init__(self, actual_dim, expected_dim):
        message = "Expected dimension %s, got %s" % (
            str(expected_dim), str(actual_dim)
        )
        super(DimensionException, self).__init__(message)
