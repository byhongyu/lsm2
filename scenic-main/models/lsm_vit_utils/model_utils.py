"""Common model utilities for VIT MAE based Large Sensor Models."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils

# Can be 'h' or 'w' assuming an input of shape [n_batch, h, w, img_c]
DEFAULT_TIME_DIM = 'h'


######################
# Data Augmentations
######################
def single_axis_augment_data(
    *,
    rng: jnp.ndarray,
    inputs: jnp.ndarray,
    augmentations: str,
    aug_dim: str = DEFAULT_TIME_DIM,
    ignored_indices: Optional[list[int]] = None,
    max_std: float = 0.5,
):
  """Applies augmentations along a single axis to a batch of sensor data images.

  This function can be added to jit compiled jax models. This should not be used
  for augmentations that are applied during dataloading of a TFDS dataset.
  Augmentations can be any of ['flip', 'stretch', 'noise']. Flip applies a
  random flip along the aug_dim axis. Stretch applies a random stretch along
  the aug_dim axis as well as a random translation along the same axis. Noise 
  adds random Gaussian noise to the image with a random standard deviation
  [0, 0.5]. This function assumes that inputs are of shape
  [n_batch, h, w, img_c] and applies augmentations independently to each batch
  sample.

  Args:
    rng: The random number key.
    inputs: Batch of sensor data images of shape [n_batch, h, w, img_c].
    augmentations: List of augmentations to apply. Augmentations can be any of
      ['flip', 'stretch', 'noise'].
    aug_dim: The dimension along which to apply the augmentations. Can be one of
      ['h', 'w'].
    ignored_indices: A list of indices, along the second spatial dimension
      (not aug_dim) to ignore when applying flip and noise
      augmentations. Helpful for temporal features or zero padding.
    max_std: The maximum standard deviation of the noise to added.

  Returns:
    The augmented batch of sensor data images, of equal shape as the inputs.
  """
  inputs = jnp.array(inputs)
  n_batch = inputs.shape[0]
  keys = jax.random.split(rng, n_batch)

  # Iterate througb batch images
  for i in range(n_batch):
    flip_rng, stretch_rng, noise_rng = jax.random.split(keys[i], 3)
    img = inputs[i]
    img_shape = img.shape

    # Time flip.
    # DO NOT TIME FLIP TIME FEATURES.
    if 'flip' in augmentations:
      flip = jax.random.bernoulli(flip_rng)
      flip_idx = jnp.full(img_shape, flip)

      if aug_dim == 'h':
        flip_idx = flip_idx.at[:, ignored_indices, :].set(0.0)
        img = jnp.where(flip_idx, jnp.flip(img, axis=0), img)
      else:
        flip_idx = flip_idx.at[:, :, ignored_indices].set(0.0)
        img = jnp.where(flip_idx, jnp.flip(img, axis=1), img)

    # Stretch.
    if 'stretch' in augmentations:
      translate_rng, stretch_rng = jax.random.split(stretch_rng)
      if aug_dim not in ['h', 'w']:
        raise ValueError(f'Unknown aug_dim: {aug_dim}')

      # Scaling.
      # NOTE: Do not change minval from 1. Otherwise 0-padding will occur.
      stretch = jax.random.uniform(
          stretch_rng, shape=(), minval=1.0, maxval=2.0
      )
      if aug_dim == 'h':
        scale = jnp.array([stretch, 1.0, 1.0])  # stretch factor in H
        len_dim = img_shape[0]
      else:
        scale = jnp.array([1.0, stretch, 1.0])  # stretch factor in W
        len_dim = img_shape[1]

      # Translation.
      new_len_dim = jnp.array(len_dim * stretch, int)
      max_translate = new_len_dim - len_dim
      shift = jax.random.randint(
          translate_rng, shape=(), minval=0, maxval=max_translate
      )
      if aug_dim == 'h':
        translate = jnp.array([-1*shift, 0, 0])  # translation in H
      else:
        translate = jnp.array([0, -1*shift, 0])  # translation in W

      # Apply stretch.
      img = jax.image.scale_and_translate(
          img, shape=img_shape, spatial_dims=[0, 1, 2],
          scale=scale, translation=translate,
          method='linear', antialias=True
      )

    # Add noise.
    # DO ADD NOISE TO TIME FEATURES
    if 'noise' in augmentations:
      add_noise_rng, std_rng, noise_rng = jax.random.split(noise_rng, 3)
      add_noise = jax.random.bernoulli(add_noise_rng)
      noise_idx = jnp.full(img_shape, add_noise)
      if aug_dim == 'h':
        noise_idx = noise_idx.at[:, ignored_indices, :].set(0.0)
      else:
        noise_idx = noise_idx.at[:, :, ignored_indices].set(0.0)

      noise = jax.random.normal(noise_rng, shape=(img_shape), dtype=jnp.float32)
      std = jax.random.uniform(std_rng, shape=(), minval=0.0, maxval=max_std)
      img = jnp.where(noise_idx, img + (std * noise), img)

    # Overwrite inputs
    inputs = inputs.at[i, :, :, :].set(img)

  return inputs


######################
# Masking Strategies
######################
# Adapted from scenic/projects/mfp/model_utils.py
def get_forecast_mask_indices(
    n_batch: int,
    n_h: int,
    n_w: int,
    n_dim_masked: int,
    mask_dim: str,
):
  """Returns forecasting indices to use for masking in MAE.

  This method generates a forecasting mask for an input of shape
  [n_batch, n_tokens, embedding_dim] where n_tokens = n_h * n_w. The mask is
  generated by randomly selecting a fraction of the height dimension to be
  masked. The mask is applied to the input tensor by setting the embedding
  vectors of the masked tokens to a fixed mask token vector.

  NOTE: mask_dim refers to the dimension along which the mask is applied for a
  2D grid of tokens (patches). The mask can be though of being applied to a set
  of patches of shape [n_batch, n_h, n_w], where n_h * n_w = n_tokens.

  Args:
    n_batch: The batch size of the sequence to generate.
    n_h: The number of (patch) tokens in the height dimension.
    n_w: The number of (patch) tokens in the width dimension.
    n_dim_masked: The number of rows or columns to mask (depending on mask_dim).
    mask_dim: The dimension along which the mask is applied. Must be one of
      ['h', 'w'].

  Returns:
    Three arrays. masked_indices of shape [n_batch, n_masked], unmasked_indices
    of shape [n_batch, n_tokens - n_masked] and binary_mask of shape
    [n_batch, n_tokens] where 1 indicates that the token is masked.
  """
  if mask_dim not in ['h', 'w']:
    raise ValueError(f'Unsupported mask_dim: {mask_dim}')

  n_tokens = n_h * n_w  # number of tokens

  if mask_dim == 'h':
    n_dim_tokens = n_h
    n_offdim_tokens = n_w
  else:
    n_dim_tokens = n_w
    n_offdim_tokens = n_h

  n_dim_unmasked = n_dim_tokens - n_dim_masked
  n_masked = n_dim_masked * n_offdim_tokens
  n_unmasked = n_dim_unmasked * n_offdim_tokens
  n_masked_batch = n_masked * n_batch  # number masked patches
  n_unmasked_batch = n_unmasked * n_batch  # number unmasked patches

  # Generate binary mask
  binary_mask = jnp.zeros((n_batch, n_tokens))
  binary_mask = jnp.reshape(binary_mask, [n_batch, n_h, n_w])
  batch_indices = jnp.arange(n_batch).reshape(n_batch, 1)

  if mask_dim == 'h':  # forecast along h dim
    binary_mask = binary_mask.at[batch_indices, -1*n_dim_masked:, :].set(1.0)
  else:  # forecast along w dim
    binary_mask = binary_mask.at[batch_indices, :, -1*n_dim_masked:].set(1.0)

  binary_mask = jnp.reshape(binary_mask, [n_batch, n_tokens])

  # Get masked/unmasked indices.
  _, mask_token = jnp.where(binary_mask == 1.0, size=n_masked_batch)
  mask_indices = jnp.reshape(mask_token, [n_batch, n_masked])
  _, unmask_token = jnp.where(binary_mask == 0.0, size=n_unmasked_batch)
  unmasked_indices = jnp.reshape(unmask_token, [n_batch, n_unmasked])

  return mask_indices, unmasked_indices, binary_mask


# Adapted from scenic/projects/mfp/model_utils.py
def get_imputation_mask_indices(
    n_batch: int,
    n_h: int,
    n_w: int,
    n_dim_masked: int,
    mask_dim: str,
    rng: jnp.ndarray,
    batch_mask: bool = False,
):
  """Returns imputation indices to use for masking in MAE.

  NOTE: mask_dim refers to the dimension along which the mask is applied for a
  2D grid of tokens (patches). The mask can be though of being applied to a set
  of patches of shape [n_batch, n_h, n_w], where n_h * n_w = n_tokens.

  Args:
    n_batch: The batch size of the sequence to generate.
    n_h: The number of (patch) tokens in the height dimension.
    n_w: The number of (patch) tokens in the width dimension.
    n_dim_masked: The number of rows or columns to mask (depending on mask_dim).
    mask_dim: The dimension along which the mask is applied. Must be one of
      ['h', 'w'].
    rng: The random number key.
    batch_mask: Whether to apply the same random mask to all batch samples.

  Returns:
    Three arrays. masked_indices of shape [n_batch, n_masked], unmasked_indices
    of shape [n_batch, n_tokens - n_masked] and binary_mask of shape
    [n_batch, n_tokens] where 1 indicates that the token is masked.
  """
  if mask_dim not in ['h', 'w']:
    raise ValueError(f'Unsupported mask_dim: {mask_dim}')

  n_tokens = n_h * n_w  # number of tokens

  if mask_dim == 'h':
    n_dim_tokens = n_h
    n_offdim_tokens = n_w
  else:
    n_dim_tokens = n_w
    n_offdim_tokens = n_h

  n_dim_unmasked = n_dim_tokens - n_dim_masked  # num unmasked patches along dim
  n_masked = n_dim_masked * n_offdim_tokens  # total num patched tokens
  n_unmasked = n_dim_unmasked * n_offdim_tokens  # total num unmasked tokens
  n_masked_batch = n_masked * n_batch  # number masked patches
  n_unmasked_batch = n_unmasked * n_batch  # number unmasked patches

  # Generate binary mask
  binary_mask = jnp.zeros((n_batch, n_tokens))
  binary_mask = jnp.reshape(binary_mask, [n_batch, n_h, n_w])

  # Apply the SAME random mask to all batch samples.
  if batch_mask:
    start_idx = jax.random.randint(
        rng, shape=(), minval=0, maxval=n_dim_unmasked
    )
    if mask_dim == 'h':
      mask = jnp.ones((n_batch, n_dim_masked, n_offdim_tokens))
      binary_mask = jax.lax.dynamic_update_slice(
          binary_mask, mask, (0, start_idx, 0)
      )
    else:
      mask = jnp.ones((n_batch, n_offdim_tokens, n_dim_masked))
      binary_mask = jax.lax.dynamic_update_slice(
          binary_mask, mask, (0, 0, start_idx)
      )
  # Apply a different random mask to all batch samples.
  else:
    if mask_dim == 'h':
      mask = jnp.ones((1, n_dim_masked, n_offdim_tokens))
    else:
      mask = jnp.ones((1, n_offdim_tokens, n_dim_masked))

    rngs = jax.random.split(rng, num=n_batch)
    for i, start_rng in enumerate(rngs):
      start_idx = jax.random.randint(
          start_rng, shape=(), minval=0, maxval=n_dim_unmasked
      )
      if mask_dim == 'h':
        binary_mask = jax.lax.dynamic_update_slice(
            binary_mask, mask, (i, start_idx, 0)
        )
      else:
        binary_mask = jax.lax.dynamic_update_slice(
            binary_mask, mask, (i, 0, start_idx)
        )

  # Reshape back to original shape.
  binary_mask = jnp.reshape(binary_mask, [n_batch, n_tokens])

  # Get masked/unmasked indices.
  _, mask_token = jnp.where(binary_mask == 1.0, size=n_masked_batch)
  mask_indices = jnp.reshape(mask_token, [n_batch, n_masked])
  _, unmask_token = jnp.where(binary_mask == 0.0, size=n_unmasked_batch)
  unmasked_indices = jnp.reshape(unmask_token, [n_batch, n_unmasked])

  return mask_indices, unmasked_indices, binary_mask


def get_random_bar_mask_indices(
    n_batch: int,
    n_h: int,
    n_w: int,
    n_dim_masked: int,
    mask_dim: str,
    rng: jnp.ndarray,
):
  """Returns structured bar masking indices to use for masking in MAE.

  Specifically, this masking strategy masks n_dim_masked random columns/rows
  (depending on mask_dim)

  Args:
    n_batch: The batch size of the sequence to generate.
    n_h: The number of (patch) tokens in the height dimension.
    n_w: The number of (patch) tokens in the width dimension.
    n_dim_masked: The number of rows or columns to mask (depending on mask_dim).
    mask_dim: The dimension along which the mask is applied. Must be one of
      ['h', 'w'].
    rng: The random number key.

  Returns:
    Three arrays. masked_indices of shape [n_batch, n_masked], unmasked_indices
    of shape [n_batch, n_tokens - n_masked] and binary_mask of shape
    [n_batch, n_tokens] where 1 indicates that the token is masked.
  """
  if mask_dim not in ['h', 'w']:
    raise ValueError(f'Unsupported mask_dim: {mask_dim}')

  # Calculate masking constants.
  if mask_dim == 'h':
    n_dim_tokens = n_h
    n_offdim_tokens = n_w
  else:
    n_dim_tokens = n_w
    n_offdim_tokens = n_h

  n_tokens = n_h * n_w  # number of tokens
  n_dim_unmasked = n_dim_tokens - n_dim_masked  # num unmasked patches along dim
  n_masked = n_dim_masked * n_offdim_tokens  # total num patched tokens
  n_unmasked = n_dim_unmasked * n_offdim_tokens  # total num unmasked tokens
  n_masked_batch = n_masked * n_batch  # number masked patches per batch
  n_unmasked_batch = n_unmasked * n_batch  # number unmasked patches per batch

  # Generate binary mask
  binary_mask = jnp.zeros((n_batch, n_tokens))
  binary_mask = jnp.reshape(binary_mask, [n_batch, n_h, n_w])  # reshape to img

  def mask_samples(key, binary_mask):
    mask_indices = jax.random.choice(
        key, jnp.arange(n_dim_tokens), shape=(n_dim_masked,), replace=False
    )
    if mask_dim == 'h':
      binary_mask = binary_mask.at[mask_indices, :].set(1)
    else:
      binary_mask = binary_mask.at[:, mask_indices].set(1)
    return binary_mask

  # Add masked indices.
  keys = jax.random.split(rng, n_batch)  # random key per batch sample
  binary_mask = jax.vmap(mask_samples, in_axes=(0, 0))(keys, binary_mask)
  # Reshape back to original shape.
  binary_mask = jnp.reshape(binary_mask, [n_batch, n_tokens])

  # Get masked/unmasked indices.
  _, mask_token = jnp.where(binary_mask == 1.0, size=n_masked_batch)
  mask_indices = jnp.reshape(mask_token, [n_batch, n_masked])
  _, unmask_token = jnp.where(binary_mask == 0.0, size=n_unmasked_batch)
  unmasked_indices = jnp.reshape(unmask_token, [n_batch, n_unmasked])

  return mask_indices, unmasked_indices, binary_mask


def get_random_partial_bar_mask_indices(
    n_batch: int,
    n_h: int,
    n_w: int,
    n_dim_masked: int,
    n_offdim_masked: int,
    mask_dim: str,
    rng: jnp.ndarray,
):
  """Returns structured partial bar masking indices to use for masking in MAE.

  Specifically, this masking strategy masks a contiguous n along the mask_dim
  dimension, and masks a random set of n_off_dim on the other dimension.

  Args:
    n_batch: The batch size of the sequence to generate.
    n_h: The number of (patch) tokens in the height dimension.
    n_w: The number of (patch) tokens in the width dimension.
    n_dim_masked: The number of rows or columns to mask (depending on mask_dim).
    n_offdim_masked: The number of columns or rows to mask along the
      off-dimension(depending on mask_dim).
    mask_dim: The dimension along which the mask is applied. Must be one of
      ['h', 'w'].
    rng: The random number key.

  Returns:
    Three arrays. masked_indices of shape [n_batch, n_masked], unmasked_indices
    of shape [n_batch, n_tokens - n_masked] and binary_mask of shape
    [n_batch, n_tokens] where 1 indicates that the token is masked.
  """
  if mask_dim not in ['h', 'w']:
    raise ValueError(f'Unsupported mask_dim: {mask_dim}')

  # Calculate masking constants.
  if mask_dim == 'h':
    n_dim_tokens = n_h
    n_offdim_tokens = n_w
  else:
    n_dim_tokens = n_w
    n_offdim_tokens = n_h

  n_tokens = n_h * n_w  # number of tokens
  n_masked = n_dim_masked * n_offdim_masked  # total num patched tokens
  n_unmasked = n_tokens - n_masked  # total num unmasked tokens
  n_masked_batch = n_masked * n_batch  # number masked patches per batch
  n_unmasked_batch = n_unmasked * n_batch  # number unmasked patches per batch

  # Generate binary mask
  binary_mask = jnp.zeros((n_batch, n_tokens))
  binary_mask = jnp.reshape(binary_mask, [n_batch, n_h, n_w])  # reshape to img

  def mask_samples(dim_key, offdim_key, binary_mask):
    mask_dim_indices = jax.random.choice(
        dim_key, jnp.arange(n_dim_tokens), shape=(n_dim_masked,), replace=False
    )
    start_idx = jax.random.randint(
        offdim_key,
        shape=(),
        minval=0,
        maxval=(n_offdim_tokens - n_offdim_masked)
    )
    # end_idx = start_idx + n_offdim_masked
    full_range = jnp.arange(n_offdim_tokens)
    mask_offdim_indices = jax.lax.dynamic_slice(
        full_range, (start_idx,), (n_offdim_masked,)
    )

    if mask_dim == 'h':
      mask_dim_indices = jnp.expand_dims(mask_dim_indices, axis=-1)
      mask_offdim_indices = jnp.expand_dims(mask_offdim_indices, axis=0)
      binary_mask = binary_mask.at[mask_dim_indices, mask_offdim_indices].set(1)
    else:
      mask_dim_indices = jnp.expand_dims(mask_dim_indices, axis=0)
      mask_offdim_indices = jnp.expand_dims(mask_offdim_indices, axis=-1)
      binary_mask = binary_mask.at[mask_offdim_indices, mask_dim_indices].set(1)
    return binary_mask

  # Add masked indices.
  # random key per batch sample in both dim and off dim
  dim_rng, offdim_rng = jax.random.split(rng, 2)
  dim_keys = jax.random.split(dim_rng, n_batch)
  offdim_keys = jax.random.split(offdim_rng, n_batch)
  binary_mask = jax.vmap(mask_samples, in_axes=(0, 0, 0))(
      dim_keys, offdim_keys, binary_mask
  )
  # Reshape back to original shape.
  binary_mask = jnp.reshape(binary_mask, [n_batch, n_tokens])

  # Get masked/unmasked indices.
  _, mask_token = jnp.where(binary_mask == 1.0, size=n_masked_batch)
  mask_indices = jnp.reshape(mask_token, [n_batch, n_masked])
  _, unmask_token = jnp.where(binary_mask == 0.0, size=n_unmasked_batch)
  unmasked_indices = jnp.reshape(unmask_token, [n_batch, n_unmasked])

  return mask_indices, unmasked_indices, binary_mask


######################
# Other
######################
def calculate_patched_img_shape(
    dataset: dataset_utils.Dataset,
    config: ml_collections.ConfigDict,
) -> Tuple[int, int]:
  """Calculate the number of patches per axis (height / width)."""
  _, height, width, _ = dataset.meta_data['input_shape']
  # legacy - add support for this to retain LSM V1 compatibility.
  # p_h, p_w = config.model.patches.size
  p_h, p_w = config.model.patcher_config.patchsize
  n_h = height // p_h
  n_w = width // p_w
  return n_h, n_w
