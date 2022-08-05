import matplotlib.pyplot as plt
import tensorflow as tf

from aol_code.plotting import draw_history_plots
from aol_code.plotting import savefig

from typing import Dict, List


class SaveMetricsPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history_dict: Dict[str: List] = {}

    def on_epoch_end(self, epoch, logs=None):
        for key in logs.keys():
            if key not in self.history_dict.keys():
                self.history_dict[key] = []
            self.history_dict[key].append(logs[key])

        draw_history_plots(self.history_dict)
        savefig("metric_plots.png")
        plt.close("all")
