
from framework.general_code.factory import Factory

from framework.data.processing_functions.color_augmentation import ColorAugmentation
from framework.data.processing_functions.make_one_hot import MakeOneHot
from framework.data.processing_functions.spatial_augmentation import SpatialAugmentation


data_dict = {
    "ColorAugmentation": ColorAugmentation,
    "MakeOneHot": MakeOneHot,
    "SpatialAugmentation": SpatialAugmentation,
}

processing_functions_factory = Factory(name="Preprocessing Functions",
                                       products=data_dict)
