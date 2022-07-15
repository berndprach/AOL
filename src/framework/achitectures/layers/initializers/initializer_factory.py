"""
(Pseudo) Factory (as in Factory Pattern) for initializers.
Can produce standard keras initializers as well as custom ones.
"""

import tensorflow.keras.initializers as initializers

from framework.general_code.factory import Factory

from framework.achitectures.layers.initializers.convolutional_center_initializers \
    import OrthogonalCenterInitializer, IdentityCenterInitializer
from framework.achitectures.layers.initializers.repeated_orthogonal_initializer \
    import RepeatedOrthogonalInitializer

from typing import Dict, Type


keras_initializer_dict = {name: getattr(initializers, name) for name in dir(initializers) if name[0].isalpha()}
# This looks something like: {
#   'Constant': <class 'keras.initializers.initializers_v2.Constant'>,
#   'GlorotNormal': <class 'keras.initializers.initializers_v2.GlorotNormal'>,
#   ...,
#   'Zeros': <class 'tensorflow.python.ops.init_ops_v2.Zeros'>}

keras_initializer_factor = Factory(name="Keras Initializers",
                                   products=keras_initializer_dict)


initializer_dict: Dict[str, Type[initializers.Initializer]] = {
    "identity_center": IdentityCenterInitializer,
    "orthogonal_center": OrthogonalCenterInitializer,
    "repeated_orthogonal": RepeatedOrthogonalInitializer,
}

initializer_factory = Factory(name="Initializers",
                              products=initializer_dict)
initializer_factory.add_subfactory(keras_initializer_factor, safe_add=True)
