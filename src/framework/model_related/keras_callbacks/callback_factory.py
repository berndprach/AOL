"""
Factory that produces tf.keras.callbacks.
"""

from framework.general_code.factory import Factory

from framework.model_related.keras_callbacks.full_test_callback import FullTestCallback
from framework.model_related.keras_callbacks.log_model_loss_callback import LogModelLossCallback
from framework.model_related.keras_callbacks.log_time_callback import LogTimeCallback
from framework.model_related.keras_callbacks.plot_layer_outputs_callback import PlotLayerOutputsCallback
from framework.model_related.keras_callbacks.print_pro_callback import PrintProCallback
from framework.model_related.keras_callbacks.save_layer_variance_plot_callback import SaveLayerVariancePlotCallback
from framework.model_related.keras_callbacks.save_metrics_plot_callback import SaveMetricsCallback
from framework.model_related.keras_callbacks.save_model_summary_callback import SaveModelSummaryCallback
from framework.model_related.keras_callbacks.stop_at_nan_callback import StopAtNaNCallback


keras_callbacks = {
    "FullTestCallback": FullTestCallback,
    "LogModelLossCallback": LogModelLossCallback,
    "LogTimeCallback": LogTimeCallback,
    "PlotLayerOutputsCallback": PlotLayerOutputsCallback,
    "PrintProCallback": PrintProCallback,
    "SaveLayerVariancePlotCallback": SaveLayerVariancePlotCallback,
    "SaveMetricsCallback": SaveMetricsCallback,
    "SaveModelSummaryCallback": SaveModelSummaryCallback,
    "StopAtNaNCallback": StopAtNaNCallback,
}

callback_factory = Factory(name="keras_callbacks", products=keras_callbacks)
