
class Model(object):

    def __init__(self, arch, mixins):
        self.arch = arch
        self.mixins = {}

        for mixin in mixins:
            if isinstance(mixin, type):
                mixin = mixin()
            self.mixins[mixin.name] = mixin

        for name, mixin in self.mixins.iteritems():
            mixin.setup(self)
            setattr(self, name, mixin.func)

    def has_mixin(self, name):
        return name in self.mixins

    def get_mixin(self, name):
        return self.mixins[name]

    def __str__(self):
        return str(self.arch)
