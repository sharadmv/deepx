
class ShapeException(Exception):

    def __init__(self, node):
        message = "Cannot infer shape for %s because shape_in is undefined." % (
            node
        )
        super(ShapeException, self).__init__(message)

class DimensionException(Exception):

    def __init__(self, node_in, node_out):
        message = "Shape mismatch between %s and %s. Expected in: %s. Inferred in: %s" % (
            node_in,
            node_out,
            str(node_out.shape_in),
            str(node_in.shape_out),
        )
        super(DimensionException, self).__init__(message)
