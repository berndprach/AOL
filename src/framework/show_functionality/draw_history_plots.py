"""
Functionality that generates plots
from tf.keras history objects.
"""

import matplotlib.pyplot as plt

from typing import Dict, List, Callable, Optional, Any

# ADDITIONAL_PLOTTING_FUNCTIONS: Dict[str, plotting_function_type] ... defined at the end of the file.
history_type = Dict[str, List[float]]
plotting_function_type = Callable[[history_type], Any]
LOG_METRIC_NAMES = ["loss", "var", "Xent", "model_loss"]


def draw_history_plots(history: history_type) -> None:
    metric_names = [key for key in history.keys() if not key.startswith("val_")]
    log_metric_names = [key for key in metric_names if key in LOG_METRIC_NAMES]

    nrof_plots = len(metric_names) \
                 + len(log_metric_names) \
                 + len(ADDITIONAL_PLOTTING_FUNCTIONS)
    nrof_columns = 3
    nrof_rows = nrof_plots // nrof_columns + (nrof_plots % nrof_columns > 0)
    fig, axs = plt.subplots(nrof_rows, nrof_columns, figsize=(12, nrof_rows * 4))  # nrof rows, nrof columns
    axs_list = ([ax for ax_row in axs for ax in ax_row] if nrof_rows > 1 else axs)
    axs_iter = iter(axs_list)

    for metric_name in metric_names:
        plt.axes(next(axs_iter))
        draw_metric(metric_name=metric_name,
                    train_values=history[metric_name],
                    val_values=history.get("val_" + metric_name, None)
                    )
        plt.legend()

        if metric_name in log_metric_names:
            plt.axes(next(axs_iter))
            draw_log_metric(history, metric_name=metric_name)
            plt.legend()

    for additional_plotting_function in ADDITIONAL_PLOTTING_FUNCTIONS.values():
        plt.axes(next(axs_iter))
        additional_plotting_function(history)
        plt.legend()

    while True:
        try:
            plt.axes(next(axs_iter))
            plt.axis("off")
        except StopIteration:
            break

    plt.tight_layout()


def draw_metric(metric_name: str,
                train_values: List = None,
                val_values: Optional[List] = None
                ):
    plt.title(metric_name.replace("_", " ").title())
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)

    plt.plot(
        [epoch_nr + 0.5 for epoch_nr in range(len(train_values))],
        train_values,
        label=f"train {metric_name}"
    )
    if val_values is not None:
        if len(val_values) < len(train_values):
            raise NotImplementedError("Validation data has different length than training data. Not implemented yet.")

        plt.plot(
            [epoch_nr + 1 for epoch_nr in range(len(val_values))],
            val_values,
            label=f"val {metric_name}"
        )


def draw_log_error(history: history_type):
    if "acc" not in history.keys(): return

    minimal_value = 1e-4
    train_error = [max(1 - acc, minimal_value) for acc in history["acc"]]

    val_error = None
    if "val_acc" in history.keys():
        val_error = [max(1 - acc, minimal_value) for acc in history["val_acc"]]

    draw_metric(metric_name="error",
                train_values=train_error,
                val_values=val_error
                )
    plt.title("Error (log scale)")
    plt.yscale("log")

    # Draw line indicating minimum value that is plotted:
    if min(train_error) < 1.01 * minimal_value or min(val_error) < 1.01 * minimal_value:
        plt.axhline(y=minimal_value, color="black", label=str(minimal_value))


def draw_log_metric(history: history_type, metric_name: str):
    if metric_name not in history.keys(): return

    draw_metric(metric_name=metric_name,
                train_values=history[metric_name],
                val_values=history.get(f"val_{metric_name}", None)
                )
    plt.title(f"{metric_name.title()} (log scale)")
    plt.yscale("log")


ADDITIONAL_PLOTTING_FUNCTIONS: Dict[str, plotting_function_type] = {
    "log_error": draw_log_error,
}
