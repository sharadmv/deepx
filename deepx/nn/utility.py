import inspect
from ..core import Node, Data

def shape_identity(shapes_in):
    return [s.copy() for s in shapes_in]

class Lambda(Node):

    def __init__(self, func, shape_func=shape_identity):
        super(Lambda, self).__init__()
        self.func = func
        self.shape_func = shape_func

    def get_outputs(self, *inputs, **kwargs):
        raw_inputs = [d.get_placeholder() for d in inputs]
        raw_outputs = self.shape_func(*raw_inputs)
        outputs = []
        for raw_output, shape_out in zip(raw_outputs, self.get_shapes_out()):
            outputs.append(Data(shape_out, placeholder=raw_output))
        return outputs

    def get_graph_inputs(self):
        return []

    def get_graph_parameters(self):
        return []

    def get_graph_updates(self, **kwargs):
        return []

    def reset_states(self):
        return

    def reset_state(self, i):
        return

    def initialize(self, **kwargs):
        return

    def reinitialize(self, **kwargs):
        return

    # Shape inference

    def set_shapes_in(self, shapes_in):
        self.shapes_in = shapes_in

    def set_shapes_out(self, shapes_out):
        self.shapes_out = shapes_out

    def get_shapes_in(self):
        return self.shapes_in

    def get_shapes_out(self):
        return self.shapes_out

    def get_num_inputs(self):
        return len(inspect.getargspec(self.func).args)

    def get_num_outputs(self):
        shapes_out = self.get_shapes_out()
        if shapes_out is None:
            return None
        return len(shapes_out)

    def infer_shape(self):
        shapes_in = self.get_shapes_in()
        if shapes_in is not None:
            if len(shapes_in) != self.get_num_inputs():
                raise TypeError("Wrong number of inputs to Lambda.")
            self.set_shapes_out(self.shape_func(shapes_in))
