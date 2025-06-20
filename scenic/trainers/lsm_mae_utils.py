"""LSM MAE Utilities.

Adapted from google3/third_party/py/scenic/projects/multimask/trainer.py.

The below includes training and eval step functions, as well as some helper
functions for processing data batches (e.g., computing targets) and getting
encoder representations from the model.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from scenic.model_lib.layers import nn_ops
# To register the preprocessing ops
from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]],
    Dict[str, Tuple[float, int]],
]
LossFn = Callable[
    [jnp.ndarray, Batch, Optional[jnp.ndarray], jnp.ndarray, bool], float
]
LrFns = Dict[str, Callable[[Any], Any]]
Patch = Union[Tuple[int, int], Tuple[int, int, int]]


def get_targets(batch: Batch, config: ml_collections.ConfigDict) -> jnp.ndarray:
  targets_type = config.masked_feature_loss.targets_type
  if targets_type == 'rgb':
    return get_rgb_targets(
        batch['input_signal'], tuple(config.model.patcher_config.patchsize)
    )
  elif targets_type == 'tokens':
    return nn.one_hot(
        batch['input_signal'], num_classes=config.model.vocab_size
    )
  else:
    raise ValueError(f'Unknown targets_type {targets_type}')


def get_rgb_targets(
    inputs: jnp.ndarray,
    patch_size: Patch,
    reconstruct_grayscale: Optional[bool] = False,
    standardise_per_patch: Optional[bool] = False,
) -> jnp.ndarray:
  """Get RGB targets to use for feature regression.

  Here, the targets are the raw rgb patches of the image.

  Args:
    inputs: Tensor of shape [b, h, w, c] or [b, t, h, w, c]. The former are
      images, and the later video.
    patch_size: The shape of the patch, defined as [ph, pw] for images, and [ph,
      pw, pt] for video.
    reconstruct_grayscale: If True, the target patch is in grayscale rather than
      rgb.
    standardise_per_patch: If true, standardise each patch by subtracting the
      mean and dividing by the standard deviation of that patch.

  Returns:
    Patched inputs. For images, shape is [b, gh * gw, ph * pw * c] where
      gh = h // ph and gw = w // pw.
      For video, shape is [b, gt * gh * gw, pt * ph * pw * c].
  """
  if inputs.ndim != 4:
    raise ValueError('Inputs should be 4D (images). Shape {inputs.shape}')

  if reconstruct_grayscale:
    # Reference for converting between RGB and grayscale.
    # https://en.wikipedia.org/wiki/Luma_%28video%29
    # Also used in tf.image.rgb_to_grayscale
    rgb_weights = jnp.tile(jnp.array([[0.2989, 0.5870, 0.1140]]), (3, 1)).T
    inputs = jnp.matmul(inputs, rgb_weights)

  assert inputs.ndim == 4, 'the input should shape BxHxWxC'
  batch = inputs.shape[0]
  # Shape is [batch, ht, wt, hp, wp, c]
  patched_image = nn_ops.patch_image(
      inputs, inputs_shape=None, patch_size=patch_size
  )
  num_tokens = patched_image.shape[1] * patched_image.shape[2]
  patched_input = jnp.reshape(patched_image, (batch, num_tokens, -1))

  if standardise_per_patch:
    patched_input = jax.nn.standardize(patched_input, axis=-1, epsilon=1e-6)

  return patched_input

# TODO(xumax, girishvn) consolidate with `mask_to_patchmask()` in dataset_utils.py
def patchify_imputationmask(
    batch: Batch,
    config: ml_collections.ConfigDict,
) -> jnp.ndarray:
  """Get patchified mask patches to align with image patches.

  Args:
    batch: A single batch of data. This batch may contain an imputation mask
      (for imputation tasks) or a NaN mask (for forecast tasks). The mask should
      be a binary tensor of shape [b, h, w, 1] where `1` indicates a missing
      pixel and `0` indicates non-missing (present).
    config: Configurations of the experiment from which to get the patch size.
      The shape of the patch, defined as [ph, pw].

  Returns:
    Patched mask. For images, shape is [b, gh * gw, ph * pw] where gh = h // ph
      and gw = w // pw. but it is flattened at the end, so [bs, h * w]
      1 indicates missingness and 0 indicates non-missing
  """

  # Get imputation mask
  if 'imputation_mask' in batch.keys() and batch['imputation_mask'] is not None:
    mask = batch['imputation_mask']
  else:
    # this should never be triggerred outside of initialization
    mask = jnp.zeros_like(batch['input_signal'])

  if mask.ndim != 4 or mask.shape[-1] != 1:
    raise ValueError(
        f'Mask should be 4D with a single channel. Got shape {mask.shape}.'
    )

  # Get patch size
  patch_size = config.model.patcher_config.patchsize

  # Apply patch_image to divide the mask into patches
  # TODO(yuzheyang): pad mask with additoinal channels before patching
  patched_mask = nn_ops.patch_image(
      mask, inputs_shape=None, patch_size=patch_size
  )

  # check shapes
  batch, height, width, dummy_channel = mask.shape
  assert batch == patched_mask.shape[0]
  assert height == patched_mask.shape[1] * patch_size[0]
  assert width == patched_mask.shape[2] * patch_size[1]
  assert dummy_channel == 1  # set to 1 because there is no RGB

  # Reshape to match the output shape of get_rgb_targets
  num_tokens = patched_mask.shape[1] * patched_mask.shape[2]
  patched_mask_flat = jnp.reshape(patched_mask, (batch, num_tokens, -1))

  assert patched_mask_flat.size == mask.size

  return patched_mask_flat


# Forked from projects/mfp/trainer.py
def representation_fn(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    representation_layer: str,
    gather_to_host: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data from the dataset.
    flax_model: A Flax model.
    representation_layer: The name of the layer to use as the representation.
    gather_to_host: Whether to gather results from all devices to the host,
      rather than leaving them distributed.

  Returns:
    Representation learned by the model for the given inputs and the labels and
    masks. If `gather_to_host` is True, these are collected from all hosts.
  """
  variables = {'params': train_state.params, **train_state.model_state}
  if 'input_signal' in batch.keys():
    _, aux = flax_model.apply(
        variables, batch['input_signal'], train=False, debug=False
    )
  elif 'inputs' in batch.keys():
    _, aux = flax_model.apply(
        variables, batch['inputs'], train=False, debug=False
    )
  else:
    raise ValueError(f'Unknown input key {batch.keys()}')
  representation = aux[representation_layer]

  if representation.ndim == 3:
    # Feature regression models return [batch, num_tokens, channels]
    logging.info(
        'Representation shape before pooling tokens: %s', representation.shape
    )
    representation = jnp.mean(representation, axis=1)
  logging.info('Representation shape: %s', representation.shape)

  if gather_to_host:
    representation = jax.lax.all_gather(representation, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  return representation, batch['label'], batch['batch_mask']


# Forked from projects/baselines/plainvit/trainer.py
def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    loss_fn: LossFn,
    lr_fns: LrFns,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
) -> Tuple[
    train_utils.TrainState, Dict[str, Tuple[float, int]], Dict[str, Any]
]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Note that in this code, the buffers of the first (train_state) and second
  (batch) arguments are donated to the computation.

  Args:
    train_state: The state of training including the current global_step,
      model_state, rng, params, and optimizer. The buffer of this argument can
      be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    loss_fn: A loss function that given logits, a batch, and parameters of the
      model calculates the loss.
    lr_fns: The learning rate fns used for the optimizer in train_state.
    metrics_fn: A metrics function that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.

  Returns:
    Updated state of training and computed metrics and some training logs.
  """
  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  # Add prediction targets
  batch['targets'] = get_targets(batch, config)
  batch['patched_imputationmask'] = patchify_imputationmask(batch, config)

  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}

    if config.masker_config.on_cpu:
      mask_indices = batch['mask_indices']
      unmasked_indices = batch['unmasked_indices']
      token_mask = batch['token_mask']
      attn_mask = batch['attn_mask']
    else:
      mask_indices, unmasked_indices, token_mask, attn_mask = (
          None,
          None,
          None,
          None,
      )

    (logits, aux), new_model_state = flax_model.apply(
        variables,
        batch['input_signal'],
        mask_indices=mask_indices,
        unmasked_indices=unmasked_indices,
        token_mask=token_mask,
        attn_mask=attn_mask,
        mutable=['batch_stats'],
        train=True,
        rngs={'dropout': dropout_rng},
        debug=debug,
    )
    loss = loss_fn(
        logits,
        batch,
        variables['params'],
        aux['token_mask'],
        config.get('loss_ignore_imputation', default=False),
    )

    return loss, (new_model_state, logits, aux['token_mask'])

  # Compute gradients with loss function.
  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (train_cost, (new_model_state, logits, masks)), grad = compute_gradient_fn(
      train_state.params
  )
  del train_cost

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grad = jax.lax.pmean(grad, axis_name='batch')
  # if config.get('max_grad_norm', None) is not None:
  #   grad = clip_grads(grad, config.max_grad_norm)
  if train_state.tx is not None:
    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)
  else:
    raise ValueError('train_state.tx is None')

  training_logs['l2_grads'] = jnp.sqrt(
      sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grad)])
  )
  ps = jax.tree_util.tree_leaves(new_params)
  training_logs['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
  us = jax.tree_util.tree_leaves(updates)
  training_logs['l2_updates'] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))
  for name, lr_fn in lr_fns.items():
    lr_name = 'learning_rate' if name == 'all' else f'learning_rate_{name}'
    training_logs[lr_name] = lr_fn(train_state.global_step)

  metrics = metrics_fn(logits, masks, batch)

  # Update the train state.
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng,
  )

  return new_train_state, metrics, training_logs


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    metrics_fn: MetricFn,
    config: ml_collections.ConfigDict,
    train: bool = True,
    debug: Optional[bool] = False,
    rng: Optional[jnp.ndarray] = None,
) -> Tuple[Dict[str, Tuple[float, int]], jnp.ndarray, Dict[str, Any]]:
  """Runs a single step of training.

  Note that in this code, the buffer of the second argument (batch) is donated
  to the computation.

  Assumed API of metrics_fn is:
  ```metrics = metrics_fn(logits, batch)
  where batch is yielded by the batch iterator, and metrics is a dictionary
  mapping metric name to a vector of per example measurements. eval_step will
  aggregate (by summing) all per example measurements and divide by the
  aggregated normalizers. For each given metric we compute:
  1/N sum_{b in batch_iter} metric(b), where  N is the sum of normalizer
  over all batches.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, params and optimizer state. The buffer of
      this argument can be donated to the computation.
    batch: A single batch of data. a metrics function, that given logits and
      batch of data, calculates the metrics as well as the loss.
    flax_model: A Flax model.
    metrics_fn: A metrics function, that given logits and batch of data,
      calculates the metrics as well as the loss.
    config: Configurations of the experiment.
    train: Whether should be called in train mode. If False, the lsm mae model
      produces an embedding as opposed to a reconstruction. If True, the lsm mae
      model produces a reconstruction (e.g. masking and decoding are enabled).
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.
    rng: Optional jax rng key.

  Returns:
    Calculated metrics, logits and aux.
  """
  # Add prediction targets
  batch['targets'] = get_targets(batch, config)
  batch['patched_imputationmask'] = patchify_imputationmask(batch, config)

  if rng is None:
    # Always use the same seed, so that eval is as consistent as possible
    rng = jax.random.PRNGKey(config.rng_seed)

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  variables = {'params': train_state.params, **train_state.model_state}

  if config.masker_config.on_cpu:
    mask_indices = batch['mask_indices']
    unmasked_indices = batch['unmasked_indices']
    token_mask = batch['token_mask']
    attn_mask = batch['attn_mask']
  else:
    mask_indices, unmasked_indices, token_mask, attn_mask = (
        None,
        None,
        None,
        None,
    )
  logits, aux = flax_model.apply(
      variables,
      batch['input_signal'],
      mask_indices=mask_indices,
      unmasked_indices=unmasked_indices,
      token_mask=token_mask,
      attn_mask=attn_mask,
      mutable=False,
      train=train,  # if True then masking and decoding in MAE are enabled.
      rngs={'dropout': dropout_rng},
      debug=debug,
  )
  metrics = metrics_fn(logits, aux['token_mask'], batch)
  return metrics, logits, aux


def restore_from_train_state(
    train_state: Any, restored_train_state: Any, strict: bool = True
) -> Any:
  """Restore train state from restored checkpoint train state.

  Args:
    train_state: Model train state to restore.
    restored_train_state: Model train state to restore from.
    strict: If strict = True, the operation ensures that the parameter KEYS
      match exactly and that the shapes are similar (it is possible that one
      param dict has a leading processes dimension).

  Returns:
    train_state with restored parameters.
  """

  def _restore_params(params, restored_params, strict):
    """Recursively updates params with restored_params."""
    # Iterate through model parameters.
    for key, value in params.items():

      # If key value is a param dictionary.
      if isinstance(value, dict):
        if key in restored_params and isinstance(restored_params[key], dict):
          _restore_params(
              value, restored_params[key], strict=strict
          )  # Recurse.
        else:
          if strict:
            raise ValueError(
                'keys do not exactly match in original and checkpointed train'
                ' states'
            )

      # If key value is a tensor.
      else:
        if key in restored_params:  # If key in restored params
          params_shape = params[key].shape
          restored_shape = restored_params[key].shape

          # Transferable shape (same shape).
          if params_shape == restored_shape:
            params[key] = restored_params[key]

          # Transferable shape (restored params have leading process dim).
          elif params_shape == restored_shape[1:]:
            params[key] = restored_params[key][0]

          # Non-transferable shape (different shape).
          else:
            raise ValueError(
                f'Unable to restore {key} from restored params of shape'
                f'{restored_shape} to params of shape {params_shape}'
            )
        else:
          if strict:
            raise ValueError(
                'keys do not exactly match in original and checkpointed train'
                ' states'
            )

  # Get parameters from trainstate and unfreeze (to traverse and modify).
  params = flax.core.unfreeze(train_state.params)
  restored_params = flax.core.unfreeze(restored_train_state.params)

  # Restore params from restored_params (done in-place).
  _restore_params(params, restored_params, strict=strict)
  if strict:
    assert compare_nested_dict_keys(params, restored_params)

  # Update train_state parameters and return.
  train_state = train_state.replace(params=flax.core.freeze(params))  # pytype: disable=attribute-error
  return train_state


def compare_nested_dict_keys(dict1, dict2, parent_key=''):
  """Compare all keys of two nested dictionaries and ensure they are exactly equal.

  Args:
      dict1 (dict): The first nested dictionary.
      dict2 (dict): The second nested dictionary.
      parent_key (str): The base key for nested levels (used for error
        messages).

  Returns:
      bool: True if all keys are exactly equal, False otherwise.
  """
  keys1 = set(dict1.keys())
  keys2 = set(dict2.keys())

  # Check for key differences in both directions
  diff1 = keys1 - keys2
  diff2 = keys2 - keys1

  if diff1 or diff2:
    print(f"Key differences at '{parent_key}':")
    if diff1:
      print(f'  In first but not in second: {diff1}')
    if diff2:
      print(f'  In second but not in first: {diff2}')
    return False

  # Recursively check nested dictionaries
  for key in keys1:
    value1 = dict1[key]
    value2 = dict2[key]

    # If both values are dictionaries, recursively check their keys
    if isinstance(value1, dict) and isinstance(value2, dict):
      new_parent_key = f'{parent_key}.{key}' if parent_key else key
      if not compare_nested_dict_keys(value1, value2, new_parent_key):
        return False
    # If one is a dict and the other isn't, the structures are not the same
    elif isinstance(value1, dict) or isinstance(value2, dict):
      print(f"Type mismatch at key '{parent_key}.{key}'")
      return False

  return True
