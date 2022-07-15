
from abc import abstractmethod

from framework.model_related.metrics.metric import Metric


class LossFunction(Metric):
    """
    A function mapping a label_batch and a prediction_batch
    to a batch of values.
    """

    name = "LossFunction"  # Used for printing.
    percentable = False

    @abstractmethod
    def __call__(self, label_batch, prediction_batch):
        pass
