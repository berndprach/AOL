
import tensorflow as tf

from framework.model_related.keras_callbacks.callback_factory import callback_factory

from framework.location_manager import location_manager as loc


def get_keras_callbacks(train_ds, lr_drops):

    # Define tensorflow callbacks (e.g. for plotting metric values):
    slvp_input = next(iter(train_ds))[0]  # (batch of training images)
    save_layer_variance_plot_callback = callback_factory.create("SaveLayerVariancePlotCallback",
                                                                input_image_batch=slvp_input),

    def lr_scheduler(epoch, current_learning_rate):
        if epoch in lr_drops:
            loc.print_pro(f"Dropping learning rate by a factor of 10! (from {current_learning_rate})")
            new_learning_rate = current_learning_rate/10
            return new_learning_rate
        else:
            return current_learning_rate

    callbacks = [
        callback_factory.create("LogModelLossCallback"),
        callback_factory.create("LogTimeCallback"),
        callback_factory.create("SaveModelSummaryCallback"),
        callback_factory.create("PrintProCallback"),
        callback_factory.create("SaveMetricsCallback"),
        save_layer_variance_plot_callback,
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
    ]

    return callbacks
