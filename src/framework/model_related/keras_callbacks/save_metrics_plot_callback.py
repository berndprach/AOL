
import tensorflow as tf

from framework.show_functionality.draw_history_plots import draw_history_plots
from framework.show_functionality.functionality import savefig

# from framework.location_manager import location_manager as loc
from typing import Dict, List


class SaveMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history_dict: Dict[str: List] = None

    def on_epoch_end(self, epoch, logs=None):
        if self.history_dict is None:
            self.history_dict = {key: [] for key in logs.keys()}

        for key in logs.keys():
            if key not in self.history_dict.keys():  # keys not occuring at every epoch.
                self.history_dict[key] = []
            self.history_dict[key].append(logs[key])

        draw_history_plots(self.history_dict)
        savefig("history_plots.png", subdirectory="metric_plots")
