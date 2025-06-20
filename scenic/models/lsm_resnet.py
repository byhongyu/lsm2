"""Implementation of Metadata Encoded ResNet model.

This model mirrors the ResNet implementation here:
third_party/py/scenic/projects/baselines/resnet.py

With the addition of optionally encoding pre-vectorized metadata and
concatenating it to the representation before the output projection.

NOTE: This is in large part a copy of the original ResNet implementation.
girishvn made the following changes.
1. added metadata encoding to MetadataEncodedResNet.
2. changed init_from_train_state to load all parameters (previsioly did not
load output projection layer)
3. Addded kwargs to model __call__ functions to make the training API more
flexible across various models.

TODO(girishvn): Modularize ResNet and MetadataEncodedResNet to avoid code
duplication. In so doing, have encode_metadata be set in the config, used to
initialize the model, and then queried externally to define the input spec.
"""

import functools
from typing import Callable, Any, Optional, Union, Dict

from absl import logging
import flax
import flax.linen as nn
from jax.nn import initializers
import jax.numpy as jnp
import ml_collections
from scenic.common_lib import debug_utils
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.layers import nn_layers
from scenic.projects.baselines import resnet


class ResNet(nn.Module):
  """ResNet architecture.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    num_filters: Num filters.
    num_layers: Num layers.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Data type, e.g. jnp.float32.
  """
  num_outputs: Optional[int]
  num_filters: int = 64
  num_layers: int = 50
  kernel_init: Callable[..., Any] = initializers.lecun_normal()
  bias_init: Callable[..., Any] = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      train: bool = False,
      debug: bool = False,
      **kwargs,
  ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Whether it is training or not.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.
      **kwargs: Additional keyword arguments. Used to make the training API more
        flexible across various models.

    Returns:
       Un-normalized logits.
    """
    if self.num_layers not in resnet.BLOCK_SIZE_OPTIONS:
      raise ValueError('Please provide a valid number of layers')
    block_sizes, bottleneck = resnet.BLOCK_SIZE_OPTIONS[self.num_layers]
    x = nn.Conv(
        self.num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=self.dtype,
        name='stem_conv')(
            x)
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        name='init_bn')(
            x)
    x = nn_layers.IdentityLayer(name='init_relu')(nn.relu(x))
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
    x = nn_layers.IdentityLayer(name='stem_pool')(x)

    residual_block = functools.partial(
        resnet.ResidualBlock, dtype=self.dtype, bottleneck=bottleneck)
    representations = {'stem': x}
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        filters = self.num_filters * 2**i
        x = residual_block(filters=filters, strides=strides)(x, train)
      representations[f'stage_{i + 1}'] = x

    # Head.
    if self.num_outputs:
      x = jnp.mean(x, axis=(1, 2))
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_outputs,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name='output_projection')(
              x)
      return x
    else:
      return representations


class ResNetClassificationModel(classification_model.ClassificationModel):
  """Implemets the ResNet model for classification."""

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return ResNet(
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return resnet._get_default_configs_for_testing()  # pylint: disable=protected-access

  # NOTE(girishvn) changed this to load all parameters.
  # Previsouly, did not load output projection layer.
  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from `restored_train_state`.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a pretrained model.
      restored_model_cfg: Configuration of the model from which the
        `restored_train_state` come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    del restored_model_cfg
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      params = flax.core.unfreeze(train_state.optimizer.target)
      restored_params = flax.core.unfreeze(
          restored_train_state.optimizer.target)
    else:
      params = flax.core.unfreeze(train_state.params)
      restored_params = flax.core.unfreeze(restored_train_state.params)
    for pname, pvalue in restored_params.items():
      params[pname] = pvalue
    logging.info('Parameter summary after initialising from train state:')
    debug_utils.log_param_shapes(params)
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      return train_state.replace(
          optimizer=train_state.optimizer.replace(
              target=flax.core.freeze(params)),
          model_state=restored_train_state.model_state)
    else:
      return train_state.replace(
          params=flax.core.freeze(params),
          model_state=restored_train_state.model_state)


class MetadataEncodedResNet(nn.Module):
  """Metadata Encoded ResNet architecture.

  A ResNet model that encodes pre-vectorized metadata and concatenates it to the
  representation before the output projection.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    num_filters: Num filters.
    num_layers: Num layers.
    metadata_encoding_fn: Function to encode metadata. Currently supports
      'concat', and None.
    kernel_init: Kernel initialization.
    bias_init: Bias initialization.
    dtype: Data type, e.g. jnp.float32.
  """
  num_outputs: Optional[int]
  num_filters: int = 64
  num_layers: int = 50
  metadata_encoding_fn: Optional[str] = 'concat'
  kernel_init: Callable[..., Any] = initializers.lecun_normal()
  bias_init: Callable[..., Any] = initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      x_metadata: jnp.ndarray,
      train: bool = False,
      debug: bool = False,
      **kwargs,
  ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      x_metadata: Metadata that is encoded into an array.
      train: Whether it is training or not.
      debug: Whether the debug mode is enabled. debug=True enables model
        specific logging/storing some values using jax.host_callback.
      **kwargs: Additional keyword arguments. Used to make the training API more
        flexible across various models.

    Returns:
       Un-normalized logits.
    """
    if self.num_layers not in resnet.BLOCK_SIZE_OPTIONS:
      raise ValueError('Please provide a valid number of layers')
    block_sizes, bottleneck = resnet.BLOCK_SIZE_OPTIONS[self.num_layers]
    x = nn.Conv(
        self.num_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding=[(3, 3), (3, 3)],
        use_bias=False,
        dtype=self.dtype,
        name='stem_conv')(
            x)
    x = nn.BatchNorm(
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        name='init_bn')(
            x)
    x = nn_layers.IdentityLayer(name='init_relu')(nn.relu(x))
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])
    x = nn_layers.IdentityLayer(name='stem_pool')(x)

    residual_block = functools.partial(
        resnet.ResidualBlock, dtype=self.dtype, bottleneck=bottleneck)
    representations = {'stem': x}
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        filters = self.num_filters * 2**i
        x = residual_block(filters=filters, strides=strides)(x, train)
      representations[f'stage_{i + 1}'] = x

    # Head.
    if self.num_outputs:
      x = jnp.mean(x, axis=(1, 2))
      # Concat metadata to the representation before the output projection.
      if self.metadata_encoding_fn is not None:
        if self.metadata_encoding_fn == 'concat':
          x = jnp.concatenate([x, x_metadata], axis=-1)
        else:
          raise ValueError(
              'Unsupported metadata encoding function: '
              f'{self.metadata_encoding_fn}'
          )

      x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_outputs,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype,
          name='output_projection')(
              x)
      return x
    else:
      return representations


class MetadataEncodedResNetClassificationModel(
    classification_model.ClassificationModel
):
  """Implemets the Metadata Encoded ResNet model for classification."""

  def build_flax_model(self) -> nn.Module:

    # Note that metadata is encoded.
    # Used in trainer to define model input spec.
    self.encode_metadata = True

    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    return MetadataEncodedResNet(
        num_outputs=self.dataset_meta_data['num_classes'],
        num_filters=self.config.num_filters,
        num_layers=self.config.num_layers,
        metadata_encoding_fn=self.config.metadata_encoding_fn,
        dtype=model_dtype)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return resnet._get_default_configs_for_testing()  # pylint: disable=protected-access

  # NOTE(girishvn) changed this to load all parameters.
  # Previsouly, did not load output projection layer.
  def init_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from `restored_train_state`.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a pretrained model.
      restored_model_cfg: Configuration of the model from which the
        `restored_train_state` come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    del restored_model_cfg
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      params = flax.core.unfreeze(train_state.optimizer.target)
      restored_params = flax.core.unfreeze(
          restored_train_state.optimizer.target)
    else:
      params = flax.core.unfreeze(train_state.params)
      restored_params = flax.core.unfreeze(restored_train_state.params)
    for pname, pvalue in restored_params.items():
      params[pname] = pvalue
    logging.info('Parameter summary after initialising from train state:')
    debug_utils.log_param_shapes(params)
    if hasattr(train_state, 'optimizer'):
      # TODO(dehghani): Remove support for flax optim.
      return train_state.replace(
          optimizer=train_state.optimizer.replace(
              target=flax.core.freeze(params)),
          model_state=restored_train_state.model_state)
    else:
      return train_state.replace(
          params=flax.core.freeze(params),
          model_state=restored_train_state.model_state)
