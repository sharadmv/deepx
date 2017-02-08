from abc import abstractmethod
from .. import T
from ..core import Shape
from .layer import ShapedLayer

__all__ = ["RecurrentLayer"]


class RecurrentLayer(ShapedLayer):

    def __init__(self, shape_in=None, shape_out=None, stateful=False, **kwargs):
        super(RecurrentLayer, self).__init__(**kwargs)
        if shape_out is not None:
            pass
        elif shape_in is not None and shape_out is None:
            shape_in, shape_out = None, shape_in
        if shape_in is not None:
            self.set_shapes_in([Shape([None, None, shape_in], batch=True, sequence=True)])
        if shape_out is not None:
            self.set_shapes_out([Shape([None, None, shape_out], batch=True, sequence=True)])
        self.stateful = stateful
        self.states = None

    def forward(self, X, **kwargs):
        if not self.get_shapes_in()[0].is_sequence():
            raise TypeError("Cannot pass non-sequence into recurrent layer.")
        return self.recurrent_forward(X)

    def recurrent_forward(self, X, **kwargs):
        states = self.get_initial_states(input_data=X)
        def step(X, states):
            result = self.step(X, states, **kwargs)
            return result
        outputs, states = T.rnn(step, X, states)
        if self.stateful:
            self.updates = zip(self.states, states)
        return outputs

    @abstractmethod
    def step(self, X, states, **kwargs):
        pass

    def create_initial_state(self, input_data, stateful, shape_index=1):
        batch_size = T.shape(input_data)[1]
        dim_out = self.get_dim_out()
        if stateful:
            if not isinstance(batch_size, int):
                raise TypeError("batch_size must be set for stateful RNN.")
            return [T.variable(T.zeros((batch_size, dim_out)))]
        return [T.alloc(0, (batch_size, dim_out), unbroadcast=shape_index)]

    def get_initial_states(self, input_data=None, shape_index=1):
        if self.states is not None:
            return self.states
        states = self.create_initial_state(input_data, self.stateful, shape_index=shape_index)
        if self.stateful:
            self.states = states
        return states

    def reset_states(self):
        if self.states is not None:
            for i, _ in enumerate(self.states):
                self.reset_state(i)

    def reset_state(self, i):
        if self.states is not None:
            T.set_value(self.states[i], T.get_value(self.states[i]) * 0)
