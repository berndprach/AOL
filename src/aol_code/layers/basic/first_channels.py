
from aol_code.layers.layer import Layer


class FirstChannels(Layer):
    """ Keeps the first few channels and ignores the rest. """

    def __init__(self, nrof_channels, ndim=4):
        super().__init__()
        self.nrof_channels = nrof_channels
        self.ndim = ndim

    def call(self, x, *args):
        if self.ndim == 4:
            return x[:, :, :, :self.nrof_channels]
        elif self.ndim == 3:
            return x[:, :, :self.nrof_channels]
        elif self.ndim == 2:
            return x[:, :self.nrof_channels]
        else:
            raise NotImplementedError
