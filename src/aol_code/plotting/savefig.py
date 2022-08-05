
import os
import matplotlib.pyplot as plt

from aol_code.location_manager import location_manager as loc


def savefig(filename, subdirectory="."):
    """ Saves current figure to file, and makes sure the folder exists. """
    if loc.outfolder_name is not None:
        directory = os.path.join(loc.outfolder_name, subdirectory)
        if subdirectory != ".":
            os.makedirs(directory, exist_ok=True)

        plt.savefig(os.path.join(directory, filename))


