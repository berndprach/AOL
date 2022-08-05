from aol_code.models import (
    get_patchwise_model,
    get_fully_fully_connected_model,
    get_convolutional_model,
    get_standard_convolutional_model,
)

from aol_code.location_manager import location_manager as loc


def get_model(model_name,
              size_parameter,
              kernel_regularizer,
              nrof_classes):
    loc.print_pro(f"Loading {model_name} model.")

    if model_name == "patch":
        model = get_patchwise_model(
            filters_after_downsize=size_parameter,
            kernel_regularizer=kernel_regularizer,
            nrof_classes=nrof_classes
        )
    elif model_name == "ffc":
        model = get_fully_fully_connected_model(
            kernel_regularizer=kernel_regularizer,
            nrof_classes=nrof_classes)
    elif model_name == "conv":
        model = get_convolutional_model(
            base_width=size_parameter,
            kernel_regularizer=kernel_regularizer,
            nrof_classes=nrof_classes
        )
    elif model_name == "std_conv":
        model = get_standard_convolutional_model(
            kernel_regularizer=kernel_regularizer,
            nrof_classes=nrof_classes)
    else:
        raise ValueError(f"Model {model_name} unknown!")

    return model
