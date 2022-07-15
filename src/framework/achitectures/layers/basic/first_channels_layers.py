
from framework.achitectures.layers.layer import Layer


class FirstChannelsLayer(Layer):
    """ Keeps the first few channels and ignores the rest. """

    print_name = "First Channels Layer"

    def __init__(self, nrof_channels, ndim=4):
        super().__init__()
        self.nrof_channels = nrof_channels
        self.ndim = ndim

    def __str__(self):
        return f"I am a {self.print_name}! (-> {self.nrof_channels} channels)"

    def call(self, x, *args):
        if self.ndim == 4:
            return x[:, :, :, :self.nrof_channels]
        elif self.ndim == 3:
            return x[:, :, :self.nrof_channels]
        elif self.ndim == 2:
            return x[:, :self.nrof_channels]
        else:
            raise NotImplementedError
