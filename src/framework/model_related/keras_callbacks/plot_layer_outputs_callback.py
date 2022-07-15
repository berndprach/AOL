import matplotlib.pyplot as plt
import tensorflow as tf

from framework.show_functionality.plotting.draw_node import draw_node_in_new_figure
from framework.show_functionality.functionality import savefig

from framework.location_manager import location_manager as loc


class PlotLayerOutputsCallback(tf.keras.callbacks.Callback):
    def __init__(self, input_image, interval_epochs=10):
        super().__init__()
        self.input_image = input_image
        self.node_model = None
        self.interval_epochs = interval_epochs
        self.layer_names = None

        plt.imshow(input_image)
        savefig(f"input_image.png", subdirectory="layer_outputs")
        loc.print_pro(f"Saved input image (of shape {input_image.shape}) to file.")

    def on_train_begin(self, logs=None):

        # loc.print_pro("Creating a model to print the outputs of the following layers:")
        # loc.print_pro(", ".join(layer.name for layer in self.model.layers))

        # layer_output_nodes = [layer.output for layer in self.model.layers]
        # self.node_model = tf.keras.models.Model(self.model.input, layer_output_nodes)
        return

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval_epochs != 0:
            return

        if self.node_model is None:
            self.node_model = create_node_model(self.model)
            self.layer_names = [layer.name for layer in self.model.layers]

        # loc.print_pro("Creating a model to print the outputs of the following layers: \n")
        # loc.print_pro("\n".join(layer.name for layer in self.model.layers))

        # layer_output_nodes = [layer.output for layer in self.model.layers]
        # # loc.print_pro(f"About to create model ")
        # self.node_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_output_nodes)

        # loc.print_pro("Plotting layer outputs!")
        # layer_outputs = self.node_model.predict(self.input_image[None, :, :, :])
        layer_outputs = self.node_model(self.input_image[None, :, :, :], training=True)
        for i, layer_output in enumerate(layer_outputs):
            # loc.print_pro(f"Plotting layer {i} output. (shape: {layer_output.shape}, dtype: {layer_output.dtype})")
            # loc.print_pro(layer_output)
            layer_output = tf.cast(layer_output, tf.float32)
            draw_node_in_new_figure(layer_output, f"Layer {i} Output")
            draw_node_in_new_figure(layer_output, f"{self.layer_names[i]} Output")
            savefig(f"layer_{i:02d}_output.png", subdirectory="layer_outputs")
            plt.close("all")


def create_node_model(original_model):
    loc.print_pro("Creating a model to print the outputs of the following layers:")
    loc.print_pro(", ".join(layer.name for layer in original_model.layers))

    layer_output_nodes = [layer.output for layer in original_model.layers]
    return tf.keras.models.Model(original_model.input, layer_output_nodes)
