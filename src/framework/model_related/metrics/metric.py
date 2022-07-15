"""

"""

from abc import ABC, abstractmethod
# from typing import Dict, Tuple, Optional, Callable, List
from typing import Any
TensorRank1 = Any


class Metric(ABC):
    """
    A function mapping a label_batch and a prediction_batch
    to a batch of values.
    """
    name = "metric"  # This one is actually important, metric will be tracked using name.
    percentable = True  # Value will be shown as {100*m}%.

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, label_batch, prediction_batch) -> TensorRank1:
        pass

