
import tensorflow.keras.layers as layers
from framework.general_code.factory import Factory

# Keras Layers seem to start with a capital letter, use that to define a dict:
keras_layers_dict = {name: getattr(layers, name) for name in dir(layers) if name[0].isupper()}
# This looks something like: {
#   'AbstractRNNCell': <class 'tensorflow.python.keras.layers.recurrent.AbstractRNNCell'>,
#   'Activation': <class 'tensorflow.python.keras.layers.core.Activation'>,
#   'ActivityRegularization': <class 'tensorflow.python.keras.layers.core.ActivityRegularization'>,
#   'Add': <class 'tensorflow.python.keras.layers.merge.Add'>,
# ... }

keras_layer_factory = Factory(name="Keras Layers",
                              products=keras_layers_dict)
