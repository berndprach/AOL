import sys
import tensorflow as tf

import aol_code.keras_callbacks as keras_callbacks
import aol_code.metrics as metrics

from aol_code.get_run_setting import get_run_setting
from aol_code.get_tensorflow_dataset import get_tensorflow_dataset
from aol_code.get_model import get_model
from aol_code.location_manager import location_manager as loc

# Parse Run ID and initialize location manager:
run_nr = int(sys.argv[1]) if len(sys.argv) > 1 else 0
loc.initialize(name="AOL", run_nr=run_nr)
rs = get_run_setting(run_nr)

# Load data, convert it to a tensorflow dataset and apply preprocessing:
train_ds, val_ds = get_tensorflow_dataset(dataset_name=rs.dataset,
                                          train_size=rs.train_dataset_size,
                                          val_size=0,
                                          nrof_classes=rs.nrof_classes,
                                          batch_size=rs.batch_size,
                                          use_testset=True)

model = get_model(model_name=rs.model_name,
                  size_parameter=rs.size_parameter,
                  kernel_regularizer=rs.weight_decay_coefficient,
                  nrof_classes=rs.nrof_classes)

# Define loss function and metrics:
offset_xent = metrics.OffsetXentFromScores(offset=rs.loss_offset,
                                           temperature=rs.loss_temperature)

metrics = [
    "acc",
    offset_xent,
    metrics.CertifiedRobustAccuracyFromScore(maximal_perturbation=36 / 255),
    metrics.CertifiedRobustAccuracyFromScore(maximal_perturbation=72 / 255),
    metrics.CertifiedRobustAccuracyFromScore(maximal_perturbation=108 / 255),
    metrics.CertifiedRobustAccuracyFromScore(maximal_perturbation=1.),
]

callbacks = [
    keras_callbacks.PrintProCallback(print_epoch_interval=10),
    keras_callbacks.SaveMetricsPlotCallback(),
    keras_callbacks.LearningRateSchedulerCallback(rs.lr_drops),
]

opt = tf.keras.optimizers.SGD(
    learning_rate=rs.lr,
    momentum=0.9,
    nesterov=True,
)

model.compile(
    optimizer=opt,
    loss=offset_xent,
    metrics=metrics
)

loc.print_pro("About to start training!", with_time=True)
history = model.fit(
    train_ds,
    epochs=rs.nrof_epochs,
    validation_data=val_ds,
    verbose=0,  # no printing (done using callbacks).
    callbacks=callbacks,
)
loc.print_pro("Finished training! Got the following results:", with_time=True)
for metric_name, metric_history in history.history.items():
    if metric_name.startswith("val_acc") or metric_name.startswith("val_cra"):
        result = metric_history[-1]
        metric_name = metric_name.replace("val_", "Validation ")
        loc.print_pro(f"{metric_name.title()}: {100 * result:.1f}%")

loc.print_pro(f"Find outputs in folder {loc.outfolder_name}.")
