
import os
import matplotlib.pyplot as plt

from framework.location_manager import location_manager as loc


def imshow(tensor, with_monocolorbar=False, **kwargs):
    if len(tensor.shape) == 2:
        im = plt.imshow(tensor, **kwargs)
        if with_monocolorbar:
            plt.colorbar()
        return im
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 1:
        return plt.imshow(tensor[:, :, 0], **kwargs)
    elif len(tensor.shape) == 3 and tensor.shape[-1] in (3, 4):
        return plt.imshow(tensor, **kwargs)
    else:
        raise NotImplementedError(f"Unexpected tensor shape: {tensor.shape}!")


def savefig(filename, subdirectory="."):
    """ Saves current figure to file, and makes sure the folder exists. """
    if loc.plots_directory is not None:
        directory = os.path.join(loc.plots_directory, subdirectory)
        os.makedirs(directory, exist_ok=True)

        plt.savefig(os.path.join(directory, filename))


