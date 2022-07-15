import os

# import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from framework.model_related.keras_callbacks.plot_layer_outputs_callback import create_node_model

import framework.achitectures.functionality as fun
# from framework.show_functionality.plotting.draw_node import draw_node_in_new_figure
from framework.show_functionality.functionality import savefig

from framework.location_manager import location_manager as loc


class SaveLayerVariancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, input_image_batch, interval_epochs=10):
        super().__init__()
        self.input_image_batch = input_image_batch
        self.node_model = None
        self.interval_epochs = interval_epochs
        self.layer_names = None

    def on_train_begin(self, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval_epochs != 0:
            return

        if self.node_model is None:
            self.node_model = create_node_model(self.model)
            self.layer_names = [layer.name for layer in self.model.layers]

        layer_outputs = self.node_model(self.input_image_batch, training=True)
        lines = ["Sqrt of Mean Node Variance:"]
        norms = []
        ubs_at_init = []  # upper bounds (of expectation) at initialization
        previous_size = None

        for i, layer_output in enumerate(layer_outputs):
            layer_output = tf.cast(layer_output, tf.float32)  # To deal with mixed precision.
            sqrt_mean_variance, layer_size = calculate_sqrt_mean_variance(layer_output)
            norms.append(sqrt_mean_variance)

            if i == 0:
                ubs_at_init.append(sqrt_mean_variance)
                previous_size = layer_size
            else:
                ub_reduction = tf.cast(tf.minimum(1, layer_size / previous_size), tf.float32) ** (1 / 2)
                new_ub = ubs_at_init[-1] * ub_reduction
                ubs_at_init.append(new_ub)
                previous_size = layer_size

            lines.append(f"N{i:02d}: {sqrt_mean_variance:7.3f} (ub: {ubs_at_init[-1]:5.2f}) ({self.layer_names[i]}).")

        plt.figure()
        plt.title("Sqrt of Mean Node Variance")
        draw_log2_plot(norms, ubs_at_init)
        plt.legend()

        savefig(f"sqrt_of_mean_node_variance.png")
        with open(os.path.join(loc.plots_directory, "sqrt_of_mean_node_variance.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        plt.close("all")


def calculate_sqrt_mean_variance(tensor_batch):
    # flat_tensor = np.reshape(tensor_batch, newshape=[])
    flat_tensor_batch = fun.partial_reshape(tensor_batch, newshape=[None, -1])
    layer_size = tf.shape(flat_tensor_batch)[1]
    centered_batch = flat_tensor_batch - tf.reduce_mean(flat_tensor_batch, axis=0, keepdims=True)
    variance_batch = tf.reduce_sum(centered_batch ** 2, axis=1)
    mean_variance = tf.reduce_mean(variance_batch, axis=0)
    sqrt_mean_variance = mean_variance ** (1 / 2)
    return sqrt_mean_variance, layer_size


def draw_log2_plot(norms, ubs_at_init):
    log2 = tf.math.log(2.)
    plt.plot([i for i in range(len(norms))], [tf.math.log(n) / log2 for n in norms], label="Values")
    plt.scatter([i for i in range(len(norms))], [tf.math.log(n) / log2 for n in norms])
    plt.plot(
        [i for i in range(len(ubs_at_init))],
        [tf.math.log(ub) / log2 for ub in ubs_at_init],
        color="grey",
        label="Upper bound (exp) init"
    )

    plt.xlabel("Layer")
    plt.ylabel("Log2(Norm)")
    plt.grid(which="both", axis="y")  # ... draws lines at major and minor ticks of y-axis. (?)
    # plt.legend()
