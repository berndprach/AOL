
import numpy as np
import matplotlib.pyplot as plt

from framework.show_functionality.functionality import imshow


def draw_image_with_channels(image_tensor, fig_subtitle=None, do_clip=True):
    """ Plots image as well as different channels. """
    nrof_plots = 1 + image_tensor.shape[2]
    fig, axs = plt.subplots(1, nrof_plots, figsize=(2 * nrof_plots + 4, 3))  # nrof rows, nrof columns
    if fig_subtitle is not None:
        fig.suptitle(f"{fig_subtitle}", fontsize=16)

    # Draw original image:
    plt.axes(axs[0])
    if do_clip:
        plt.title("clipped image")
        imshow(np.clip(image_tensor, 0., 1.))
    else:
        plt.title("image")
        imshow(image_tensor)

    # Draw channels:
    for j in range(1, nrof_plots):
        channel = j-1
        plt.axes(axs[j])
        plt.title(str(channel))
        imshow(image_tensor[:, :, channel])
        plt.colorbar()
