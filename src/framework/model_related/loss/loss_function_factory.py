"""
Gathers all loss functions into one directory.
We do not want to acomplish this using sideeffects,
so we instead create functions that register loss functions,
and use those to explicitely register those loss functions.
"""

from framework.general_code.directory import Directory

# Import all files that register loss functions:
from framework.model_related.loss.loss_functions.identity_loss import register_identity_loss
from framework.model_related.loss.loss_functions.xent import register_xent

# Create new directory:
loss_function_directory = Directory()

# Register all loss function to directory:
register_identity_loss(loss_function_directory)
register_xent(loss_function_directory)
