import cPickle as pickle

class Model(object):

    def __init__(self, arch, mixins):
        self.arch = arch
        self.mixins = {}

        for mixin in mixins:
            if isinstance(mixin, type):
                mixin = mixin()
            self.mixins[mixin.name] = (mixin, mixin.priority)

        for mixin, priority in sorted(self.mixins.values(), key=lambda x: x[1]):
            mixin.setup(self)
            setattr(self, mixin.name, mixin.get_function())

    def has_mixin(self, name):
        return name in self.mixins

    def get_mixin(self, name):
        return self.mixins[name][0]

    def __str__(self):
        return str(self.arch)

    def save_parameters(self, location):
        state = self.arch.get_state()
        with open(location, 'wb') as fp:
            pickle.dump(state, fp)

    def load_parameters(self, location):
        with open(location, 'rb') as fp:
            state = pickle.load(fp)
        self.arch.load_state(state)
