

import matplotlib.pyplot as plt

from framework.show_functionality.functionality import imshow


def draw_grid(tensor, fig_subtitle=None, max_size=(2, 6), with_colorbar=True):
    """
    if len(max_size) == 1: plots tensor[j, :, :, :] in subplot (1, j)
    if len(max_size) == 2: plots tensor[i, j, :, :, :] in subplot (row, col) = (i, j)
    """
    if len(max_size) == 1:
        nrof_plots = max_size[0]
        fig, axs = plt.subplots(1, nrof_plots, figsize=(2 * nrof_plots + 4, 3))  # nrof rows, nrof columns
        if fig_subtitle is not None:
            fig.suptitle(f"{fig_subtitle}", fontsize=16)
        for j in range(nrof_plots):
            plt.axes(axs[j])
            if j >= tensor.shape[0]:
                plt.axis("off")
                continue
            plt.title(str(j))
            imshow(tensor[j])
            if with_colorbar:
                plt.colorbar()

    elif len(max_size) == 2:
        nrof_rows, nrof_cols = max_size
        fig, axs = plt.subplots(nrof_rows, nrof_cols, figsize=(2 * nrof_cols + 4, 2 * nrof_rows + 1))
        if fig_subtitle is not None:
            fig.suptitle(f"{fig_subtitle}", fontsize=16)
        for i in range(nrof_rows):
            for j in range(nrof_cols):
                plt.axes(axs[i][j])
                if i >= tensor.shape[0] or j >= tensor.shape[1]:
                    plt.axis("off")
                    continue
                plt.title(f"({i}, {j})")
                imshow(tensor[i, j])
                if with_colorbar:
                    plt.colorbar()
    else:
        raise ValueError(f"Expect argument max_size to have length in [1, 2]!"
                         f"Got max_size={max_size}.")
