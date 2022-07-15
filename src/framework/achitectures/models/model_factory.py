
from framework.achitectures.models.fullly_fully_connected import get_fully_fully_connected_model
from framework.achitectures.models.convolutional import get_convolutional_model
from framework.achitectures.models.patchwise import get_patchwise_model
from framework.achitectures.models.standard_convolutional import get_standard_convolutional_model

get_model_functions = {
    "ffc": get_fully_fully_connected_model,
    "conv": get_convolutional_model,
    "patch": get_patchwise_model,
    "std_conv": get_standard_convolutional_model,
}


def get_model(name, **kwargs):
    get_model_function = get_model_functions[name]
    return get_model_function(**kwargs)
