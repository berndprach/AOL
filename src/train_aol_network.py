
import sys
import tensorflow as tf

from get_run_setting import get_run_setting
from get_keras_callbacks import get_keras_callbacks
from framework.location_manager import location_manager as loc

from framework.achitectures.models.model_factory import get_model
from framework.model_related.metrics.metric_factory import metric_directory
from framework.model_related.loss.loss_function_factory import loss_function_directory

from framework.data.load_data_factory import load_data
from framework.data.generate_tf_dataset import generate_tf_datasets
from framework.data.sequential_preprocessor import SequentialPreprocessor

# Parse run id and initialize location manager:
run_nr = int(sys.argv[1])
loc.initialize(name="Experiment", run_nr=run_nr)
rs = get_run_setting(run_nr)


# Load data, convert it to a tensorflow dataset and apply preprocessing:
data = load_data(name=rs.dataset,
                 train_set_size=rs.train_dataset_size,
                 validation_set_size=0)
tf_ds = generate_tf_datasets(data)

preprocessor = SequentialPreprocessor(preprocessor_configurations=[
    ("MakeOneHot", {"label_dim": rs.nrof_classes}),
    ("ColorAugmentation", {}),
    ("SpatialAugmentation", {}),
])
preprocessed_tf_ds = preprocessor.get_preprocessed_dataset(tf_ds, batch_size=rs.batch_size)
train_ds = preprocessed_tf_ds["train"]
val_ds = preprocessed_tf_ds["test"]  # Evaluate on the test set for the final results:


# Load model:
if rs.model_kind == "patch":
    model = get_model(name="patch",
                      filters_after_downsize=rs.size_parameter,
                      kernel_regularizer=rs.weight_decay_coefficient,
                      nrof_classes=rs.nrof_classes)
else:
    model = get_model(name=rs.model_kind,
                      kernel_regularizer=rs.weight_decay_coefficient)


# Define loss function and metrics:
offset_xent = loss_function_directory.create("offset_xent",
                                             "from_scores",
                                             offset=rs.loss_offset,
                                             temperature=rs.loss_temperature)

metrics = [
    "acc",
    metric_directory.create("certainty", "from_scores"),
    offset_xent,
    loss_function_directory.create("xent", "from_scores"),
    metric_directory.create("certified_robust_accuracy", "from_scores", maximal_perturbation=36 / 255),
    metric_directory.create("certified_robust_accuracy", "from_scores", maximal_perturbation=72 / 255),
    metric_directory.create("certified_robust_accuracy", "from_scores", maximal_perturbation=108 / 255),
    metric_directory.create("certified_robust_accuracy", "from_scores", maximal_perturbation=1.),
    metric_directory.create("variance", "from_scores"),
]

# Define callbacks for things such as learning rate scheduler and metric plotting:
callbacks = get_keras_callbacks(train_ds, lr_drops=rs.lr_drops)

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
    verbose=2,  # print one line per epoch.
    callbacks=callbacks,
)
loc.print_pro("Finished training!", with_time=True)
