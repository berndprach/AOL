"""
Gathers all metrics into one directory.
We do not want to acoplish this using sideeffects,
so we instead create functions that register metrics,
and use those to explicitely register those metrics.
"""

from framework.general_code.directory import Directory

# Import all files that register loss functions:
from framework.model_related.metrics.metric_functions.mean_certainty import register_certainty_metrics
from framework.model_related.metrics.metric_functions.prediction_variance import register_variance_metrics
from framework.model_related.metrics.metric_functions.certified_robust_accuracy import register_cra_metrics

# Create new directory:
metric_directory = Directory()

# Register all loss function to directory:
register_certainty_metrics(metric_directory)
register_variance_metrics(metric_directory)
register_cra_metrics(metric_directory)
