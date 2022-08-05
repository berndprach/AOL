import tensorflow as tf

from aol_code.location_manager import location_manager as loc


class PrintProCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_epoch_interval=1, separator=" | "):
        super().__init__()
        self.print_epoch_interval = print_epoch_interval
        self.separator = separator
        self.print_header_in = 0
        self.metric_keys = None
        self.header = None

    def on_epoch_end(self, epoch, logs=None):
        real_epoch = epoch + 1

        if self.print_header_in == 0:
            if self.header is None:
                keys = [key for key in logs.keys()
                        if not key.startswith("val_")]
                self.metric_keys = sorted(keys)
                header_strs = ["Train/Val "]
                for key in self.metric_keys:
                    header_str = key.title() + ":"
                    header_strs.append(f"{header_str:16.16}")
                self.header = self.separator.join(header_strs)

            loc.print_pro(self.header)
            self.print_header_in = 10

        if real_epoch % self.print_epoch_interval != 0: return

        self.print_header_in -= 1
        output_line = create_output_line(real_epoch,
                                         self.metric_keys,
                                         logs,
                                         self.separator)
        loc.print_pro(output_line)


def create_output_line(epoch, metric_keys, logs, separator):
    print_strs = [f"Epoch {epoch:4d}"]
    for key in metric_keys:
        train_value = logs[key]
        validation_value = logs.get(f"val_{key}", 0.)
        if is_percentable(key):
            whitespace = " "*5
            print_strs.append(f"{100 * train_value:4.1f}%/"
                              f"{100 * validation_value:4.1f}%"
                              f"{whitespace}")
        else:
            whitespace = " "*7
            print_strs.append(f"{train_value:4.3g}/"
                              f"{validation_value:4.3g}"
                              f"{whitespace}")

    return separator.join(print_strs)


def is_percentable(key):
    return (
            key.startswith("acc")
            or key.startswith("cra")
    )
