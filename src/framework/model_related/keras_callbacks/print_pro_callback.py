import tensorflow as tf

from framework.location_manager import location_manager as loc


# PERCENTABLE = ["acc", "cert"]
def is_percentable(key):
    return (
        key.startswith("acc")
        or key.startswith("cert")
        or key.startswith("cra")
    )


class PrintProCallback(tf.keras.callbacks.Callback):
    def __init__(self, print_epoch_interval=1, print_batch_interval=200):
        super().__init__()
        self.print_epoch_interval = print_epoch_interval
        self.print_batch_interval = print_batch_interval

    def on_epoch_end(self, epoch, logs=None):
        real_epoch = epoch + 1
        if real_epoch % self.print_epoch_interval != 0: return

        # keys = [key for key in logs.keys() if not key.startswith("val_")]
        # keys = sorted(keys)
        # print_strs = [f"Epoch {real_epoch}"]
        # for key in keys:
        #     train_value = logs[key]
        #     validation_value = logs.get(f"val_{key}", "0.")
        #     print_strs.append(f"{key}: {train_value:.4f} ({validation_value:.4f})")
        # # parts = [f"{key}: {logs[key]}/{logs.get('val_'+key, '-')}" for key in keys]
        # separator = "   "
        # loc.print_pro(separator.join(print_strs))

        print_to_file(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        real_batch = batch + 1
        if real_batch % self.print_batch_interval != 0: return

        # loc.print_pro(f"   ... batch {real_batch}:
        # " + "   ".join([f"{key}: {logs[key]:.4f}" for key in logs.keys()]))
        loc.print_pro(f"   ... batch {real_batch}:   " +
                      "   ".join([f"{key.title()}: {logs[key]:4.3g}" for key in logs.keys()]))


def print_to_file(epoch, logs):
    real_epoch = epoch + 1

    keys = [key for key in logs.keys() if not key.startswith("val_")]
    keys = sorted(keys)
    print_strs = [f"Epoch {real_epoch}"]
    for key in keys:
        train_value = logs[key]
        validation_value = logs.get(f"val_{key}", 0.)
        if is_percentable(key):
            print_strs.append(f"{key.title()}: {100 * train_value:5.2f}%/{100 * validation_value:5.2f}%")
        else:
            # print_strs.append(f"{key.title()}: {train_value:5.4f}/{validation_value:5.4f}")
            print_strs.append(f"{key.title()}: {train_value:4.3g}/{validation_value:4.3g}")
    # parts = [f"{key}: {logs[key]}/{logs.get('val_'+key, '-')}" for key in keys]
    separator = "   "

    loc.print_pro(separator.join(print_strs))
