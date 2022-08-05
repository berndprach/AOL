
import matplotlib.pyplot as plt

from typing import Dict, List, Optional

history_type = Dict[str, List[float]]


def draw_history_plots(history: history_type) -> None:
    metric_names = [key for key in history.keys()
                    if not key.startswith("val_")]

    nrof_plots = len(metric_names)
    nrof_columns = 3
    nrof_rows = nrof_plots // nrof_columns + (nrof_plots % nrof_columns > 0)
    fig, axs = plt.subplots(nrof_rows, nrof_columns,
                            figsize=(12, nrof_rows * 4))
    axs_list = ([ax for ax_row in axs for ax in ax_row]
                if nrof_rows > 1 else axs)
    axs_iter = iter(axs_list)

    for metric_name in metric_names:
        plt.axes(next(axs_iter))
        train_values = history[metric_name]
        val_values = history.get("val_" + metric_name, None)
        draw_metric(metric_name, train_values, val_values)
        plt.legend()

    remove_remaining_axes(axs_iter)
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
        plt.plot(
            [epoch_nr + 1 for epoch_nr in range(len(val_values))],
            val_values,
            label=f"val {metric_name}"
        )


def remove_remaining_axes(axs_iter):
    for axs in axs_iter:
        plt.axes(axs)
        plt.axis("off")
