
import tensorflow.keras.initializers as keras_initializers

import aol_code.layers.initializers as initializers

from typing import Dict, Type

custom_initializers: Dict[str, Type[keras_initializers.Initializer]] = {
    "identity_center": initializers.IdentityCenterInitializer,
    "orthogonal_center": initializers.OrthogonalCenterInitializer,
    "repeated_orthogonal": initializers.RepeatedOrthogonalInitializer,
}


def get(identifier):
    initializer = custom_initializers.get(identifier, identifier)
    # initializer = custom_initializers.get(identifier,
    #                                       keras_initializers.get(identifier))
    return initializer

