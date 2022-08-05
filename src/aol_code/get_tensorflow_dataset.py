
from aol_code.data.load_data_functions import load_cifar10, load_cifar100
from aol_code.data.generate_tf_dataset import generate_tf_datasets
from aol_code.data.sequential_preprocessor import SequentialPreprocessor
import aol_code.data.preprocessing_functions as ppf


def get_tensorflow_dataset(dataset_name,
                           train_size,
                           val_size,
                           nrof_classes,
                           batch_size,
                           use_testset=False):

    if dataset_name == "cifar10":
        data = load_cifar10(train_size=train_size,
                            val_size=val_size)
    elif dataset_name == "cifar100":
        data = load_cifar100(train_size=train_size,
                             val_size=val_size)
    else:
        raise ValueError(f"Dataset {dataset_name} unknown!")

    tf_ds = generate_tf_datasets(data)

    preprocessor = SequentialPreprocessor(preprocessors=[
        ppf.MakeLabelOneHot(label_dim=nrof_classes),
        ppf.ColorAugmentation(),
        ppf.SpatialAugmentation(),
    ])
    preprocessed_tf_ds = preprocessor(tf_ds, batch_size=batch_size)

    train_ds = preprocessed_tf_ds["train"]
    if use_testset:
        # Evaluate on the test set for the final results:
        return train_ds, preprocessed_tf_ds["test"]
    else:
        return train_ds, preprocessed_tf_ds["val"]

