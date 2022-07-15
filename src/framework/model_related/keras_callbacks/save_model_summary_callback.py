import os

import tensorflow as tf

from framework.location_manager import location_manager as loc


class SaveModelSummaryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0:
            return

        # Print model summary to file:
        model_filename = os.path.join(loc.outfolder_name, "model.txt")
        with open(model_filename, "w") as f:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda line: f.write(line + "\n"), line_length=80)



