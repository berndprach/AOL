
from framework.data.load_data_functions.load_cifar10 import load_cifar10
from framework.data.load_data_functions.load_cifar100 import load_cifar100

load_data_functions = {
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
}


def load_data(name, **kwargs):
    load_data_function = load_data_functions[name]
    return load_data_function(**kwargs)
