"""TODO: xumax - DO NOT SUBMIT without one-line documentation for masker.

TODO: xumax - DO NOT SUBMIT without a detailed description of masker.
"""

import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
import tensorflow as tf
from google3.experimental.largesensormodels.scenic.trainers.masking.masker_config import MaskStrategy_Config

def _cpu_get_random_mask_indices(
    maskstrategy_config: MaskStrategy_Config,
    maskable_mask: tf.Tensor,
    existing_mask: tf.Tensor,
    seed: int,
):
  """Generates a random binary mask while preserving existing masked tokens.

  This function calculates how many tokens to mask based on the mask probability
  from `maskstrategy_config` and the number of eligible tokens in
  `maskable_mask`. It preserves all existing masked tokens from
  `existing_mask` and randomly selects additional tokens from the unmasked
  portion of `maskable_mask` to reach the required number of masks.

  Args:
      maskstrategy_config: Configuration containing masking parameters like
        mask_probability. The mask_probability determines what percentage of
        eligible tokens should be masked.
      maskable_mask: Binary tensor indicating which tokens are eligible for
        masking (1 = maskable, 0 = non-maskable). Must be same shape as
        existing_mask.
      existing_mask: Binary tensor where 1 indicates already masked tokens that
        must remain masked. Must be same shape as maskable_mask.
      seed: Integer seed for random operations to ensure reproducibility.

  Returns:
      A binary tensor of the same shape as existing_mask where:
        1 = Masked token (including both originally masked and newly masked)
        0 = Unmasked token

  Raises:
      tf.errors.InvalidArgumentError: If the number of existing masked tokens
        exceeds the computed maximum allowed masks (based on mask probability).
  """
  # Calculate total required masked tokens and validate against existing masks
  n_maskable = tf.reduce_sum(maskable_mask)
  n_masked = int(
      tf.math.floor(
          tf.cast(n_maskable, tf.float32) * maskstrategy_config.mask_probability
      )
  )
  existing_mask_count = tf.cast(tf.reduce_sum(existing_mask), tf.int32)
  tf.debugging.assert_less_equal(
      existing_mask_count,
      n_masked,
      message=(
          f'[random] Existing mask count {existing_mask_count} exceeds the upper' 
          f'bound of total masked tokens {n_masked}'
      ),
  )

  # Identify candidate positions excluding already masked tokens
  maskable_mask = tf.cast(existing_mask == 0, tf.int32) * maskable_mask

  # Shuffle eligible positions for random selection
  shuffled_maskable_inds = tf.where(maskable_mask == 1)
  shuffled_maskable_inds = tf.random.shuffle(shuffled_maskable_inds, seed=seed)

  # Select required number of new masks
  total_remaining_to_mask = n_masked - existing_mask_count
  maskable_inds_tomask = shuffled_maskable_inds[:total_remaining_to_mask]

  # Apply new masks to existing mask tensor
  updates = tf.ones(total_remaining_to_mask, dtype=tf.int32)
  token_mask = tf.tensor_scatter_nd_update(
      tensor=existing_mask, indices=maskable_inds_tomask, updates=updates
  )
  return token_mask


def _cpu_get_bar_mask_indices(
    maskstrategy_config: MaskStrategy_Config,
    maskable_mask: tf.Tensor,
    maskable_inds_alongmaskdimidx: tf.Tensor,
    existing_mask: tf.Tensor,
    mask_dim_idx: int,
    seed: int,
):
  """Generates a structured "bar" mask while preserving existing masked tokens.

  This strategy creates rectangular mask regions by:
  1. Selecting dimensions (rows if mask_dim_idx=0, columns if mask_dim_idx=1)
     proportional to the mask probability
  2. Creating potential mask regions from selected dimensions
  3. Randomly selecting unmasked tokens within these regions to meet mask quota
  4. Combining new masks with existing ones while maintaining validity

  The process preserves all existing masked tokens from `existing_mask`.

  Args:
      maskstrategy_config: Configuration containing parameters like
        mask_probability that determines the proportion of dimensions to mask
      maskable_mask: Binary tensor (same shape as existing_mask) indicating
        eligible positions for new masks (1 = maskable, 0 = non-maskable)
      maskable_inds_alongmaskdimidx: Tensor of indices along the target
        dimension (mask_dim_idx) that belong to this masking group. For rows
        (mask_dim_idx=0), these would be row indices; for columns
        (mask_dim_idx=1), column indices
      existing_mask: Binary tensor (same shape as maskable_mask) where 1
        indicates pre-existing masked tokens that must be preserved
      mask_dim_idx: Dimension index for mask orientation (0 = row-wise bars, 1 =
        column-wise bars)
      seed: Integer seed for reproducible random operations

  Returns:
      A binary tensor of same shape as existing_mask where:
      1 = Masked token (preserved existing masks + new bar masks)
      0 = Unmasked token

  Raises:
      tf.errors.InvalidArgumentError: If existing masked tokens exceed the
          maximum allowed masks calculated from mask probability
  """
  ### TODO(xumax) Fix bar masking method such that it constructs exact number
  ### defined by mask_probability, this will enable below probability check
  ### as well as allow for better, exact maskstrictperc

  # Calculate total required masks and validate against existing masks
  n_maskable = tf.reduce_sum(maskable_mask)
  n_masked = int(
      tf.math.floor(
          tf.cast(n_maskable, tf.float32) * maskstrategy_config.mask_probability
      )
  )
  existing_mask_count = tf.cast(tf.reduce_sum(existing_mask), tf.int32)
  tf.debugging.assert_less_equal(
      existing_mask_count,
      n_masked,
      message=(
          f'[bar] Existing mask count ({existing_mask_count}) exceeds the upper'
          f' bound of total masked tokens ({n_masked})'
      ),
  )

  # identify which dims to mask out
  total_dims_to_mask = int(
      tf.math.floor(
          maskable_inds_alongmaskdimidx.shape[0]
          * maskstrategy_config.mask_probability
      )
  )
  if maskstrategy_config.mask_dim_contiguous:
    if maskstrategy_config.mask_dim_forecasting:
      # forecasting
      maskable_dims_tomask = tf.range(
          maskable_inds_alongmaskdimidx.shape[0] - total_dims_to_mask,
          maskable_inds_alongmaskdimidx.shape[0],
      )
    else:
      # contiguous temp imputation
      shuffled_maskable_dims = tf.range(
          maskable_inds_alongmaskdimidx.shape[0] - total_dims_to_mask
      )
      shuffled_maskable_dims = tf.random.shuffle(
          shuffled_maskable_dims, seed=seed
      )
      maskable_dims_tomask = tf.range(
          shuffled_maskable_dims[0],
          shuffled_maskable_dims[0] + total_dims_to_mask,
      )
  else:
    # non-contiguous temp imputation
    shuffled_maskable_dims = tf.range(maskable_inds_alongmaskdimidx.shape[0])
    shuffled_maskable_dims = tf.random.shuffle(
        shuffled_maskable_dims, seed=seed
    )
    maskable_dims_tomask = tf.gather(
        maskable_inds_alongmaskdimidx,
        tf.cast(shuffled_maskable_dims[:total_dims_to_mask], tf.int32),
    )

  # The indices for rows to update need to be of shape [N, 1]. the tf.cast(x==0) is a trick to flip 0->1 and 1->0
  indices = tf.expand_dims(
      maskable_dims_tomask, axis=tf.cast(mask_dim_idx == 0, tf.int32)
  )
  updates = tf.ones(
      (tf.shape(maskable_dims_tomask)[0], tf.shape(existing_mask)[1]),
      dtype=existing_mask.dtype,
  )
  # tomask includes the indices identified to mask out by the end
  tomask_mask = tf.scatter_nd(indices, updates, shape=existing_mask.shape)
  # now we remove the existing from maskable by subtracting it, s.t. all places
  # with existing are either 0 or -1, then relu forces -1 to 0
  tomask_mask_without_existing = tf.nn.relu(tomask_mask - existing_mask)

  # we have too many maskable inds because it now exceeds n_masked (with the existing mask)
  # so randomly choose a subset of (n_masked - existing_mask_count)
  shuffled_maskable_inds = tf.where(tomask_mask_without_existing == 1)
  shuffled_maskable_inds = tf.random.shuffle(shuffled_maskable_inds, seed=seed)
  maskable_inds_tomask = shuffled_maskable_inds[
      : (n_masked - existing_mask_count)
  ]

  # updating existing_mask with the maskable inds
  updates = tf.ones(len(maskable_inds_tomask), dtype=tf.int32)
  final_mask = tf.tensor_scatter_nd_update(
      tensor=existing_mask, indices=maskable_inds_tomask, updates=updates
  )

  return final_mask
