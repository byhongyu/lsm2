"""Utilities for processing and loading data."""

from typing import Dict

import jax.numpy as jnp
import ml_collections  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np

from google3.experimental.largesensormodels.scenic.trainers.masking.masker import _cpu_get_random_mask_indices, _cpu_get_bar_mask_indices

Batch = Dict[str, jnp.ndarray]


# TODO(xumax) modularize this function into smaller parts into masker.py
def mask_example(
    example: Batch,
    masker_config,
    patch_size: tuple[int, int],
    input_size: tuple[int, int, int],
    seed: int = 0,
):
  """Applies structured masking to input examples using configured strategies.

  Typically called in lsm_v2_pretraining_dataset.py

  Combines multiple masking strategies across different dimensions while:
  1. Respecting existing imputation masks (where 1 = already masked)
  2. Applying either random or bar-shaped masking patterns
  3. Supporting different mixing strategies for multiple mask configurations
  4. Handling strict masking percentages and attention mask construction

  Args:
      example: Input batch containing: - 'input_signal': (H, W, C) input
        features (C=1 dummy channel) - 'imputation_mask': Optional existing mask
        (1 = masked)
      masker_config: Configuration object specifying: - maskstrategy_list: Mask
        strategies to apply - mixstrategy: "within_instance" or
        "between_instance" mixing - inherited: Whether to use existing
        imputation mask - strictmaskperc: Optional percentage of tokens to
        strictly mask
      patch_size: (H, W) patch dimensions for converting to patch space
      input_size: (H, W, C) original input dimensions
      seed: Random seed for reproducible masking

  Returns:
      Modified example dict with added keys:
          - 'token_mask': Binary patch-level mask (1 = masked). This is the
          prediction_masked used in the final loss function.
          - 'mask_indices': Flat indices of masked patches. This is used for
          drop out removal.
          - 'unmasked_indices': Flat indices of unmasked patches. This is used
          for drop out removal.
          - 'attn_mask': Attention mask (1 = attend, 0 = ignore) This is a list
          of length of encoder sequence, i.e. length of the patched
          input_sequence AFTER the drop out removal.

  Raises:
      ValueError: For unsupported strategies or invalid combinations
  """

  # Handle existing imputation mask (1 = missing, 0 = present)
  if masker_config.inherited and example['imputation_mask'] is not None:
    # masker_config.inherited = True then we want the imputation_mask to be used
    # in our calculations throughout the function
    # example['imputation_mask'] is not None is just a check to ensure it exists
    imputation_mask = example['imputation_mask']
  else:
    # if inherited is false, then we dont want to be aware of existing
    # imputation_mask, pretending like none of it exists with zeros.
    imputation_mask = tf.zeros(input_size)

  # Convert pixel-level mask to patch-level mask
  patched_imputation_mask = mask_to_patchmask(
      imputation_mask, input_size=input_size, patch_size=patch_size
  )
  patched_shape = patched_imputation_mask.shape

  # Organize strategies by dimension (0=time, 1=modal)
  # this is important for the within-instance mixstrategy so that each instance can be
  # proportioned along the instance correctly
  total_mask_along_time = []
  total_mask_along_modal = []
  for maskstrat in masker_config.maskstrategy_list:
    if maskstrat.mask_dim_idx == 0:
      total_mask_along_time.append(maskstrat)
    elif maskstrat.mask_dim_idx == 1:
      total_mask_along_modal.append(maskstrat)
  if len(total_mask_along_time) > 0 and len(total_mask_along_modal) > 0:
    raise ValueError('code does not support this yet, but its close')

  if len(total_mask_along_modal) > 0:
    raise ValueError('not tested yet')

  # Initialize combined mask and process each dimension group
  token_mask = tf.zeros(patched_shape, dtype=tf.int32)
  for mask_dim_idx, total_mask_along in enumerate(
      [total_mask_along_time, total_mask_along_modal]
  ):
    # used to skip if there are no masking stragies along a given dim
    if len(total_mask_along) == 0:
      continue

    # Calculate strategy weights and normalize
    weights = np.array([strat.weight for strat in total_mask_along])
    weights = weights / np.sum(weights)

    # Handle different mixing strategies
    if masker_config.mixstrategy == 'within_instance':
      # Split dimension indices proportionally between strategies
      indices = tf.range(patched_shape[mask_dim_idx])
      shuffled_indices = tf.random.shuffle(indices, seed=seed)

      # Calculate the sizes of the splits and the last split for any rounding errors
      split_sizes = (weights * len(indices)).astype(np.int32)
      split_sizes[-1] = len(indices) - np.sum(split_sizes[:-1])
      # each instance will get a specific subset of indices along a dimension to
      # apply a given masking strategy, i.e. giving one strategy the first 12 hours
      # and then giving the other strategy the last 12 hours.
      split_indices_all = tf.split(shuffled_indices, split_sizes)
      selected_strategy_idx = None

    elif masker_config.mixstrategy == 'between_instance':
      # Select single strategy per instance using categorical sampling
      selected_strategy_idx = tf.random.categorical(
          tf.math.log(weights[None, :]), 1, seed=seed, dtype=tf.int32
      )[0, 0]
      # set to ensure compatability with within_instance
      split_indices_all = [
          tf.range(patched_shape[mask_dim_idx], dtype=tf.int32)
          for _ in range(len(weights))
      ]

    else:
      raise ValueError('Unexpected mixstrategy')

    # Apply each strategy to its allocated indices
    # split_indices_i is the subset of indices that a given strategy can mask on
    for i, (split_indices_i, maskstrategy_config) in enumerate(
        zip(split_indices_all, total_mask_along)
    ):
      # Skip unselected strategies in between_instance mode
      if selected_strategy_idx is not None and selected_strategy_idx != i:
        continue

      # Create mask for current strategy's indices
      # this is only used for the within_instance strategy
      strategy_mask = tf.zeros(patched_imputation_mask.shape, dtype=tf.int32)
      strategy_mask += tf.reduce_sum(
          tf.one_hot(
              split_indices_i,
              patched_imputation_mask.shape[mask_dim_idx],
              axis=mask_dim_idx,
              dtype=tf.int32,
          ),
          # trick to flip 0->1 and 1->0
          axis=int(mask_dim_idx == 0),
          keepdims=True,
      )

      # Combine with existing masks for this strategy region
      if maskstrategy_config.inherited_depend:
        # using existing mask add n more mask tokens until we hit the total mask
        # probability.
        region_existing_mask = patched_imputation_mask * strategy_mask
      else:
        # if we have inherited independence, then there are no mask tokens
        # present in the original imputation mask so that it does not influence
        # how the masking strategies identify indicies
        region_existing_mask = tf.zeros_like(patched_imputation_mask)

      # Apply selected masking strategy
      if maskstrategy_config.strategy == 'random':
        token_mask_temp = _cpu_get_random_mask_indices(
            maskstrategy_config=maskstrategy_config,
            existing_mask=region_existing_mask,
            maskable_mask=strategy_mask,
            seed=seed,
        )
      elif maskstrategy_config.strategy == 'bar':
        token_mask_temp = _cpu_get_bar_mask_indices(
            maskstrategy_config=maskstrategy_config,
            maskable_mask=strategy_mask,
            maskable_inds_alongmaskdimidx=split_indices_i,
            existing_mask=region_existing_mask,
            mask_dim_idx=mask_dim_idx,
            seed=seed,
        )
      else:
        raise ValueError(
            f'Mask strategy, {maskstrategy_config.strategy}, not supported'
            ' on cpu'
        )

      ### TODO(xumax) Fix bar masking method such that it constructs exact number
      ### defined by mask_probability, this will enable below probability check
      ### as well as allow for better, exact maskstrictperc

      # # Validate mask count matches expected
      # expected_masked = int(
      #     tf.math.floor(
      #         tf.cast(tf.reduce_sum(strategy_mask), tf.float32)
      #         * maskstrategy_config.mask_probability
      #     )
      # )
      # tf.debugging.assert_equal(
      #     tf.reduce_sum(token_mask_temp),
      #     expected_masked,
      #     message=(
      #         f'Mask count mismatch for {maskstrategy_config.strategy} strategy'
      #     ),
      # )

      # Aggregate masks across strategies
      token_mask += token_mask_temp

  # Final processing of mask outputs
  token_mask = tf.reshape(token_mask, [-1])  # Flatten to 1D
  # total number of patches
  n_total = tf.reduce_prod(patched_shape)

  # Construct "Drop Out Removal" indices and + "Attention Masking" attn_mask
  # this below code enables two different masking strategies, *simultaneously*

  # (1) "Drop Out Removal": this is the classical ViT mask handling method, where
  # masked out tokens are removed from the sequence before being input into the
  # encoder. Then they are added back in after the encoder as mask_tokens. This
  # strategy is nice as it helps boost computational effeciency, as less tokens
  # are processed by the encoder. However, if there is a different number of
  # mask tokens per instance, this strategy does not work because each instance
  # would have a different length, and could not be batched together to be input
  # into the encoder.
  # vit.py needs *mask_indices* and *unmasked_indices* for this strategy

  # (2) "Attention Masking": this strategy just forces the ViT's encoder's attention
  # to ignore all masked tokens via an attention mask that says 1 for missing and
  # 0 for present. This strategy is nice because it allows for a flexible number
  # of mask tokens PER INSTANCE and not a set number of mask tokens.

  # This code works by first splitting the masked out indices into inds_remaining
  # for attention masking and inds_strictmaskperc for the drop out removal
  # it is necessary for indices to not be shuffled and the attention masking indices
  # to be the first portion. This is because drop out removal happens FIRST in the pipeline
  # then the attention mask covers the the prefixed-portion of the sequence for masking.
  # Visually.....
  # Our total mask looks like this
  # [0 1 1 1 0]
  # Our input_signal sequence looks like this
  # [a b c d e]
  # inds_remaining = 1,2 | inds_strictmaskperc = 3
  # After the "drop out removal", our sequence looks like this, as it is being
  # put through the encoder
  # [a b c d]
  # then we apply our attention mask, which looks like this [note that it is
  # flipped because we attend to non-masked regions]
  # [1 0 0 1]
  # Then encoder output looks like this where Ea and Ee did not use b or c in the
  # calculations
  # [Ea Eb Ec Ee]
  # now we re-add mask tokens in for attention
  # [Ea M M Ee]
  # finally we re-add mask tokens from the drop out removal
  # [Ea M M M Ee]
  if masker_config.strictmaskperc is not None:
    patched_imputation_mask = tf.cast(
        tf.reshape(patched_imputation_mask, [-1]), tf.int32
    )
    # Create full mask by combining artifical and inherited masks
    token_mask_full = tf.cast(
        (token_mask + patched_imputation_mask) > 0, tf.int32
    )

    # Calculate actual number of masked tokens
    n_masked = tf.cast(tf.reduce_sum(token_mask_full), tf.int32)
    # Get indices of all masked tokens
    inds_token_mask_full = tf.cast(tf.where(token_mask_full == 1), tf.int32)

    # Calculate required number of tokens to meet strictmaskperc
    n_strictmaskperc = tf.cast(
        tf.math.floor(
            masker_config.strictmaskperc * tf.cast(n_total, tf.float32)
        ),
        dtype=tf.int32,
    )
    tf.debugging.assert_less_equal(
        n_strictmaskperc,
        n_masked,
        'n_masked is less than n_strictmaskperc ... somehow',
    )

    # Split masked indices into two groups:
    # - inds_strictmaskperc: Tokens to keep masked
    # - inds_remaining: Originally masked tokens to potentially unmask
    inds_remaining, inds_strictmaskperc = tf.split(
        inds_token_mask_full, [n_masked - n_strictmaskperc, n_strictmaskperc]
    )

    # Create token mask with only strictly masked tokens, this token mask will 
    # be used to generate mask_indices, which determines which tokens are 
    # "drop out removed" from the vit
    token_mask_strict = tf.scatter_nd(
        indices=inds_strictmaskperc,
        updates=tf.ones_like(inds_strictmaskperc[:, 0]),
        shape=[n_total],
    )

    # Construct attention mask logic:
    # 1. We want to attend to all tokens EXCEPT those in inds_remaining
    # 2. inds_remaining represent tokens that were originally masked but
    #    aren't part of the strictmaskperc requirement
    # 3. Create mask with 0s at inds_remaining (no attention) and 1s elsewhere

    # First create mask with 1s at inds_remaining positions
    attn_mask = tf.scatter_nd(
        indices=inds_remaining,
        updates=tf.ones_like(inds_remaining[:, 0]),
        # num of tokens left after removing from n_strictmaskperc
        shape=[n_total - n_strictmaskperc],
    )

    # Invert mask (0->1, 1->0) to create final attention mask
    # Now 1 indicates positions to attend to, 0 indicates masked positions
    attn_mask = tf.cast(attn_mask == 0, tf.int32)[None, :]

  else:
    attn_mask = None
    token_mask_full = token_mask
    token_mask_strict = token_mask

  example['token_mask'] = token_mask_full

  mask_indices = tf.squeeze(
      tf.where(token_mask_strict == 1), axis=-1
  )
  # handle case in which there are no mask_indices when strictmaskperc=0.0 by 
  # setting mask_indices to -1
  example['mask_indices'] = tf.cond(
        tf.size(mask_indices) > 0,
        lambda: mask_indices,
        lambda: tf.constant([-1], dtype=mask_indices.dtype)
  )

  # unmasked_indices is a superset of the unremoved indices from the dropout
  # removal, but also says the the "attn_mask" area is completely unmasked
  # because the attn_mask will be masked out at the encoder attn step instead
  # this works bc token_mask_strict is 0s and only 1s for dropout removal
  example['unmasked_indices'] = tf.squeeze(
      tf.where(token_mask_strict == 0), axis=-1
  )
  example['attn_mask'] = attn_mask  # pytype: disable=container-type-mismatch

  return example


# TODO(xumax, girishvn) consolidate with `patchify_imputationmask()` in lsm_mae_utils.py
def mask_to_patchmask(
    mask,
    input_size,
    patch_size,
    overlap=None,
    mechanism='absolute',
    thresh_amt=0.8,
):
  """Converts a 2D  mask into a patch-based mask with specified mechanisms for determining patch values.

  Parameters:
  ----------
  mask : jax.numpy.ndarray
    The input 2D tensor of shape (x, y, d) where d is a dummy channel of size 1,
    and x, y are the spatial dimensions of each mask in the batch.
  patch_size : tuple of int
    The size of each patch as (i, j). Each patch will have dimensions (i, j).
  mechanism : str, optional, default="absolute"
    The mechanism to determine the values of the patchmask:
    - "absolute": A patch in the mask corresponds to 1 in the patchmask
                  if all elements in the patch are 1.
    - "1threshold": A patch in the mask corresponds to 1 in the patchmask
                    if at least `thresh_amt` proportion of elements in the patch
                    are 1.
  thresh_amt : float, optional, default=0.8
    The threshold value used with the "1threshold" mechanism. Indicates the
    minimum
    proportion of elements in a patch that must be 1 for the patchmask to be set
    to 1.

  Returns:
  -------
  jax.numpy.ndarray
    A 2D tensor (patchmask) of shape `(x // i, y // j)` representing the
    aggregated patches for each batch.
    The value at each location indicates whether the corresponding patch
    satisfies the
    condition defined by the chosen mechanism.

  Raises:
  ------
  ValueError:
    If an invalid mechanism is provided.
  """
  # Extract dimensions and patch size
  x, y, _ = input_size

  i, j = patch_size

  # Reshape the mask into patches
  assert (
      x % i == 0
  ), "input mask's time dim should be evenly divisible by patch's time dim"
  assert (
      y % j == 0
  ), "input mask's feat dim should be evenly divisible by patch's feat dim"

  # Reshape into (b, a, i, b, j) where a = x // i and b = y // j
  reshaped = tf.reshape(
      mask,
      (x // i, i, y // j, j),
      # # removed this because we would like mask to be evenly divisible by patch dims
      # mask[: x - (x % i), : y - (y % j)], (x // i, i, y // j, j)
  )

  if mechanism == 'absolute':
    # Check if all elements in each patch are 1
    patchmask = tf.reduce_all(tf.equal(reshaped, True), axis=(1, 3))
  elif mechanism == '1threshold':
    # Calculate the fraction of ones in each patch
    patch_mean = tf.math.reduce_mean(tf.cast(reshaped, tf.float32), axis=(1, 3))
    # Check if at least thresh_amt proportion of the patch is 1
    patchmask = patch_mean >= tf.constant(thresh_amt, dtype=tf.float32)
  else:
    raise ValueError(f'Unknown mechanism: {mechanism}')

  return tf.cast(patchmask, tf.int32)


############################
# MAE / Vision Transformer
# Patch Compatible Resize Functions
############################
def get_height_crop_width_pad(
    feat_shape: tuple[int, int, int], patch_size: tuple[int, int]
):
  """Gets H crop, and W pad values for an image to allow for even patching.

  NOTE: This assumes that the image is of the shape [H, W, C],
  where H is the time axis, and W is the feature axis.

  Args:
    feat_shape: tuple; Shape of the image (H, W, C).
    patch_size: tuple; Size of the patches to extract from the image (H, W).

  Returns:
    crop_h: int; Number of rows to crop from the top of the image.
    pad_w: tuple; Number of columns to pad on the left and right of the image.
    feat_shape_new: tuple; Shape of the new feature image (H, W, C).
  """

  height, width, channels = feat_shape
  patch_h, patch_w = patch_size

  # Crop H (time) for even patches
  num_patches_h = height // patch_h
  crop_h = height - int(num_patches_h * patch_h)

  # Pad W to make even patches
  remainder_w = width % patch_w
  if remainder_w != 0:
    pad_total = patch_w - remainder_w
    pad1 = pad_total // 2
    pad2 = pad_total - pad1
  else:
    pad1 = 0
    pad2 = 0
  pad_w = (pad1, pad2)

  # Calculate new shape
  height_new = height - crop_h
  width_new = pad1 + width + pad2
  feat_shape_new = (height_new, width_new, channels)

  return (crop_h, 0), pad_w, feat_shape_new


def patch_compatible_resize_example(
    example: tf.Tensor,
    patch_size: tuple[int, int],
):
  """Crops and pads features to allow for a integer number of patches.

  NOTE: This assumes that the image is in the shape [H, W, C], where H is the
  Time axis which can be cropped, and W is the feature axis which can be padded.

  NOTE: This should be applied AFTER augmentations as to ensure that noise is
    not applied to zero-padding.

  Args:
    example: A dictionary of inputs containing at least the 'input_signal',
      'labels', and possibly 'datetime_signal' fields.
    patch_size: tuple; Size of the patches to extract from the image (H, W).

  Returns:
    A dictionary of inputs containing at least the 'input_signal', 'labels',
      and possibly 'datetime_signal' fields. Where 'input_signal', and possibly
      'datetime_signal' fields are H cropped and W padded.
  """
  # Parse inputs
  features = example['input_signal']
  time_features = example['datetime_signal']  # datetime features

  # Crop time axis (h) and pad feature axis (w)
  crop_h, pad_w, _ = get_height_crop_width_pad(features.shape, patch_size)
  features = features[crop_h[0] :, :, :]
  features = tf.pad(
      features,
      paddings=[[0, 0], pad_w, [0, 0]],
      mode='CONSTANT',
      constant_values=0,
  )

  # Crop time axis (h) and pad time feature axis (w) of datetime features.
  if time_features is not None:
    time_crop_h, time_pad_w, _ = get_height_crop_width_pad(
        time_features.shape, patch_size
    )
    time_features = time_features[time_crop_h[0] :, :, :]
    time_features = tf.pad(
        time_features,
        [[0, 0], time_pad_w, [0, 0]],
        mode='CONSTANT',
        constant_values=0,
    )

  example['input_signal'] = features
  example['datetime_signal'] = time_features

  return example


############################
# Augmentations and Cropping
############################
def augment_example(example, augmentations, seed=0):
  """Applies augmentations (stretch, flip, noise) to the features."""

  augmented_feat = example['input_signal']
  height, width, _ = augmented_feat.shape

  # Stretch (along time/height axis).
  if 'stretch' in augmentations:
    apply_stretch = tf.random.uniform([], minval=0, maxval=1, seed=seed)
    if apply_stretch >= 0.5:
      stretch = tf.random.uniform([], minval=1.0, maxval=1.5, seed=seed + 1)
      stretched_height = int(height * stretch)
      augmented_feat = tf.image.resize(
          augmented_feat, size=[int(stretched_height), int(width)]
      )
      offset_height = stretched_height - height
      augmented_feat = tf.image.crop_to_bounding_box(
          image=augmented_feat,
          target_height=height,
          target_width=width,
          offset_height=offset_height,
          offset_width=0,
      )

      # TODO(girishvn): apply translate?
      augmented_feat = augmented_feat[
          -1 * height :, :, :
      ]  # crop to original size

  # Flip (along time/height axis).
  if 'flip' in augmentations:
    apply_flip = tf.random.uniform([], minval=0, maxval=1, seed=seed + 3)
    if apply_flip >= 0.5:
      augmented_feat = tf.image.flip_up_down(augmented_feat)

  # Noise (gaussian).
  if 'noise' in augmentations:
    apply_noise = tf.random.uniform([], minval=0, maxval=1, seed=seed + 4)
    if apply_noise >= 0.5:
      noise_std = tf.random.uniform([], minval=0.0, maxval=0.5, seed=seed + 5)
      noise = tf.random.normal(
          shape=tf.shape(augmented_feat),
          mean=0.0,
          stddev=noise_std,
          seed=seed + 6,
      )
      augmented_feat += noise

  example['input_signal'] = augmented_feat
  return example


def time_crop_example(example, patch_size, start, end):
  """Time window the input."""
  # Get valid starts and ends
  if end is None:
    end = 1
  if start is None:
    start = 0

  # Get updated patch shape.
  feature = example['input_signal']
  _, _, feat_shape_new = get_height_crop_width_pad(
      tuple(feature.shape), patch_size
  )

  # Get number of patches along time axis (h).
  p_h = patch_size[0]
  h = feat_shape_new[0]
  n_h = h // p_h

  # Time Crop image based on horizon.
  start_idx = int(start * n_h) * p_h
  end_idx = int(end * n_h) * p_h
  cropped_feature = feature[start_idx:end_idx, :, :]

  # Update and return.
  example['input_signal'] = cropped_feature
  return example


############################
# Data Filtering
############################
def filter_log_values(example, allowed_labels):
  """Filter out examples where the label is not in allowed_labels.

  Args:
    example: The input example dictionary.
    allowed_labels: A list of allowed labels.

  Returns:
    A boolean tensor indicating whether the example should be kept or not.
  """
  label = example['metadata']['log_value']
  label = tf.cast(tf.reshape(label, []), tf.int32)
  keep_example = tf.reduce_any(tf.math.equal(label, allowed_labels))
  return keep_example


def filter_imputation_ratio(example, max_imputation_ratio: tf.float32 = 0.2):
  """Filter out examples where the imputation ratio is too high.

  Args:
    example: The input example dictionary.
    max_imputation_ratio: The maximum imputation ratio to keep.

  Returns:
    A boolean tensor indicating whether the example should be kept or not.
  """
  imp_ratio = example['imputation_ratio']
  imp_ratio = tf.cast(tf.reshape(imp_ratio, []), tf.float32)
  keep_example = tf.math.less(imp_ratio, max_imputation_ratio)
  return keep_example


# NOTE: Leaving this here for future reference (girishvn).
# def filter_by_nested_value(
#     example, allowed_labels, nested_path='metadata/log_value'
# ):
#   """Filter out examples where a nested label value is not in allowed_labels.

#   NOTE: This although more general, this implementation can be upto 5x slowers
#   than the above filter_log_values function.

#   Args:
#     example: The input example dictionary.
#     allowed_labels: A list of allowed labels.
#     nested_path: A string of the nested key '/' separated path to the label
#       value.

#   Returns:
#     A boolean tensor indicating whether the example should be kept or not.
#   """
#   label = functools.reduce(dict.get, nested_path.split('/'), example)
#   label = tf.cast(tf.reshape(label, []), tf.int32)
#   keep_example = tf.reduce_any(tf.math.equal(label, allowed_labels))
#   return keep_example
