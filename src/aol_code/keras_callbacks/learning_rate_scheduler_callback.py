import tensorflow as tf

from aol_code.location_manager import location_manager as loc


class LearningRateSchedulerCallback(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, learning_rate_drop_epochs):
        def lr_scheduler(epoch, current_learning_rate):
            if epoch in learning_rate_drop_epochs:
                loc.print_pro(f"Epoch {epoch:4d}: "
                              f"Dropping learning rate by a factor of 10. "
                              f"(From {current_learning_rate:.2g}.)")
                new_learning_rate = current_learning_rate / 10
                return new_learning_rate
            else:
                return current_learning_rate

        super().__init__(lr_scheduler)
