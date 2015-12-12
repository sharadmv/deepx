from ..node import Mixin

class predict(Mixin):

    name = 'predict'

    def get_inputs(self):
        return [i.get_data() for i in self.arch.get_input()]

    def get_result(self):
        return self.arch.get_activation().get_data()
