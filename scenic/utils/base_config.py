"""Contains base_config, which all of our configs should inherit from"""

import copy
import inspect
import ml_collections


class Base_Config(ml_collections.ConfigDict):
  """A base configuration class that extends `ml_collections.ConfigDict`

  and provides automatic support for deep copying.

  This class uses `inspect` to introspect the constructor signature
  and extract both positional and keyword arguments, allowing
  `copy.deepcopy` to create a new instance without requiring
  explicit definition of copy logic in subclasses.

  Inherit from this class to create configuration objects that
  support deep copying without needing to redefine the `__deepcopy__`
  method or provide manual argument handling.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def __deepcopy__(self, memo):
    # Get the signature of the __init__ method
    signature = inspect.signature(self.__class__.__init__)
    kwargs = {}

    # Extract arguments from the instance, skipping `self`
    for name, param in signature.parameters.items():
      if name == 'self':
        continue
      if hasattr(self, name):
        # Deep copy the attribute
        kwargs[name] = copy.deepcopy(getattr(self, name), memo)

    # Create a new instance with the extracted arguments
    copied = self.__class__(**kwargs)
    memo[id(self)] = copied
    return copied

  def copy_and_resolve_references(self, visit_map=None):
    # Instead of calling the default logic that instantiates the object,
    # simply return a deep copy of the current instance.
    return copy.deepcopy(self)
