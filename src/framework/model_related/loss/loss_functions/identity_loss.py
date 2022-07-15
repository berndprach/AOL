
import tensorflow as tf

from framework.general_code.directory import Directory

from framework.model_related.loss.loss_functions.loss_function import LossFunction


class IdentityLoss(LossFunction):
    name = "IdentityLoss"

    def __call__(self, label_batch, prediction_batch):
        return tf.reduce_sum(label_batch * prediction_batch, axis=-1)


def register_identity_loss(loss_function_directory: Directory):
    loss_function_directory.register(
        IdentityLoss,
        "identity_loss",
    )


