
# import unittest
import matplotlib.pyplot as plt

from framework.show_functionality.draw_history_plots import draw_history_plots

EXAMPLE_HISTORY = {
    "loss": [1., 0.8, 0.65, 0.5],
    "acc": [0.1, 0.5, 0.8, 1.],
    "cert": [0., 0.8, 1.6, 2.4],
    "cra0.25": [0.0, 0.0, 0.1, 0.2],
    "var": [1., 2., 4., 8.],
    "val_loss": [1., 0.9, 0.8, 0.9],
    "val_acc": [0.1, 0.4, 0.6, 0.5],
    "val_cert": [0., 0.6, 0.7, 0.7],
    "val_cra0.25": [0., 0., 0., 0.],
    "val_var": [1.5, 3., 6., 12.],
}


class TestDrawHistoryPlots:
    def test_draw_history_plots(self):
        print("Plotting history plots:")
        draw_history_plots(EXAMPLE_HISTORY)
        plt.savefig("test_draw_history_plot_output.png")
        plt.show()


if __name__ == "__main__":
    print("Started Test.")
    TestDrawHistoryPlots().test_draw_history_plots()
    plt.savefig("test_draw_history_plot_output.png")
