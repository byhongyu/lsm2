"""LSM Supervised Model Utilities.

Adapted from google3/third_party/py/scenic/projects/multimask/trainer.py.

The below includes training and eval step functions, as well as some helper
functions for processing data batches. These functions are used for both
supervised baselines and for linear-probing models.
"""

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import flax.linen as nn
from flax.training import common_utils
import jax
from jax.example_libraries import optimizers as jax_optimizers
import jax.numpy as jnp
import ml_collections
import optax
from scenic.model_lib.base_models import classification_model
from scenic.model_lib.base_models import model_utils as scenic_model_utils
from scenic.train_lib import train_utils


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]],
    Dict[str, Tuple[float, int]],
]
LossFn = Callable[
    [jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]], float
]
LrFns = Dict[str, Callable[[Any], Any]]
Patch = Union[Tuple[int, int], Tuple[int, int, int]]


# Forked from projects/baselines/plainvit/trainer.py
def train_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    lr_fns: LrFns,
    loss_fn: LossFn,
    max_grad_norm: Optional[float] = None,
    config: ml_collections.ConfigDict,  # pylint: disable=unused-argument
    debug: Optional[bool] = False,
    has_aux: bool = False,
    finetune: bool = False,
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
    lr_fns: The learning rate fns used for the optimizer in train_state.
    loss_fn: Loss function.
    max_grad_norm: Maximum gradient norm to use for gradient clipping.
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during training. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.
    has_aux: A boolean of if the model returns and aux parameter.
    finetune: A boolean of if the model is in finetune mode. This only applies
      embedding models that are being linear-probed / fine-tuned.

  Returns:
    Updated state of training and computed metrics and some training logs.
  """
  # 0. Setup.
  training_logs = {}
  new_rng, rng = jax.random.split(train_state.rng)

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  # 1. Define training loss function
  def training_loss_fn(params):
    variables = {'params': params, **train_state.model_state}

    # Get model outputs.
    # NOTE: Different models have different inputs/outputs defined below:
    # Models may / may not return `aux` - auxilary dictionary of intermediaries.
    # Models may expect different inputs (as subset of those passed here).
    # For consistent API, models should be able to accept unused inputs via a
    # kwargs parameter.
    # TODO(girishvn, xumax): A more flexible model API would be useful.
    # For example, something where model-specific inputs are defined in a
    # function and then accordingly parsed out of the batch.

    # LSM ViT MAE specific inputs.
    if (
        config.model_name in ['lsm_vit_mae'] and
        config.masker_config.on_cpu
    ):
      mask_indices = batch['mask_indices']
      unmasked_indices = batch['unmasked_indices']
      token_mask = batch['token_mask']
      attn_mask = batch['attn_mask']
    else:
      mask_indices = None
      unmasked_indices = None
      token_mask = None
      attn_mask = None

    # Get model outputs.
    if has_aux:
      (logits, _), new_model_state = flax_model.apply(
          # Data inputs.
          variables=variables,
          x=batch['input_signal'],
          x_metadata=batch.get('input_metadata'),

          # LSM ViT specific inputs.
          mask_indices=mask_indices,
          unmasked_indices=unmasked_indices,
          token_mask=token_mask,
          attn_mask=attn_mask,
          finetune=finetune,

          # Generic model inputs.
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
          debug=debug,
      )
    else:
      logits, new_model_state = flax_model.apply(
          # Data inputs.
          variables=variables,
          x=batch['input_signal'],
          x_metadata=batch.get('input_metadata'),

          # LSM ViT specific inputs.
          mask_indices=mask_indices,
          unmasked_indices=unmasked_indices,
          token_mask=token_mask,
          attn_mask=attn_mask,
          finetune=finetune,

          # Generic model inputs.
          mutable=['batch_stats'],
          train=True,
          rngs={'dropout': dropout_rng},
          debug=debug,
      )

    # Compute and return loss.
    loss = loss_fn(logits, batch['label'], batch['batch_mask'])
    return loss, (new_model_state, logits)

  # 2. Compute gradients with loss function.
  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (train_cost, (new_model_state, logits)), grad = compute_gradient_fn(
      train_state.params
  )
  del train_cost

  # 3. Average gradients across devices.
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  # Optionally clip the gradients.
  grad = jax.lax.pmean(grad, axis_name='batch')
  if max_grad_norm is not None:
    grad = jax_optimizers.clip_grads(grad, max_grad_norm)

  # 4. Update parameters based on gradients using the optimizer.
  if train_state.tx is not None:
    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)
  else:
    raise ValueError('train_state.tx is None')

  # 5. Add learning statistics to training logs.
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

  # 6. Get classification metrics.
  # This includes (generally) includes loss and accuracy.
  metrics = classification_model.classification_metrics_function(
      logits, batch, target_is_onehot=True
  )

  # 7. Update the train state.
  new_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=train_state.global_step + 1,
      opt_state=new_opt_state,
      params=new_params,
      model_state=new_model_state,
      rng=new_rng,
  )

  # 8. Return the updated train state, metrics, and training logs.
  return new_train_state, metrics, training_logs


def eval_step(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    config: ml_collections.ConfigDict,
    debug: Optional[bool] = False,
    rng: Optional[jnp.ndarray] = None,
    target_is_onehot: bool = True,
    has_aux: bool = False,
) -> Tuple[Dict[str, Tuple[float, int]], Dict[str, Any]]:
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
    config: Configurations of the experiment.
    debug: Whether the debug mode is enabled during evaluation. `debug=True`
      enables model specific logging/storing some values using
      jax.host_callback.
    rng: Optional jax rng key.
    target_is_onehot: If the target is one-hot encoded.
    has_aux: A boolean of if the model returns and aux parameter.

  Returns:
    Calculated metrics, and aux.
  """
  # 0. Setup.
  if rng is None:
    # Always use the same seed, so that eval is as consistent as possible
    rng = jax.random.PRNGKey(config.rng_seed)

  # Bind the rng to the host/device we are on.
  dropout_rng = train_utils.bind_rng_to_host_device(
      rng, axis_name='batch', bind_to='device'
  )

  # 1. Get model outputs.
  # NOTE: Different models have different outputs. These are defined in
  # train_step() above. Please read the documentation there.

  # LSM ViT MAE specific inputs.
  if (
      config.model_name in ['lsm_vit_mae'] and
      config.masker_config.on_cpu
  ):
    mask_indices = batch['mask_indices']
    unmasked_indices = batch['unmasked_indices']
    token_mask = batch['token_mask']
    attn_mask = batch['attn_mask']
  else:
    mask_indices = None
    unmasked_indices = None
    token_mask = None
    attn_mask = None

  # Get model outputs.
  variables = {'params': train_state.params, **train_state.model_state}
  if has_aux:
    logits, _ = flax_model.apply(
        # Data inputs.
        variables=variables,
        x=batch['input_signal'],
        x_metadata=batch.get('input_metadata'),

        # LSM ViT specific inputs.
        mask_indices=mask_indices,
        unmasked_indices=unmasked_indices,
        token_mask=token_mask,
        attn_mask=attn_mask,

        # Generic model inputs.
        train=False,
        mutable=False,
        rngs={'dropout': dropout_rng},
        debug=debug,
    )
  else:
    logits = flax_model.apply(
        # Data inputs.
        variables=variables,
        x=batch['input_signal'],
        x_metadata=batch.get('input_metadata'),

        # LSM ViT specific inputs.
        mask_indices=mask_indices,
        unmasked_indices=unmasked_indices,
        token_mask=token_mask,
        attn_mask=attn_mask,

        # Generic model inputs.
        train=False,
        mutable=False,
        rngs={'dropout': dropout_rng},
        debug=debug,
    )

  # 2. Get classification metrics.
  metrics = classification_model.classification_metrics_function(
      logits, batch, target_is_onehot=target_is_onehot
  )

  # Get predictions, targets, and other measures to caluclate confusion matrix.
  if target_is_onehot:
    one_hot_targets = batch['label']
  else:
    one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

  targets = jnp.argmax(one_hot_targets, axis=-1)  # True classes in indices
  preds = jnp.argmax(logits, axis=-1)  # Predictions in class indices
  batch_mask = batch.get('batch_mask')
  merged_targets = jax.lax.all_gather(targets, axis_name='batch')
  merged_preds = jax.lax.all_gather(preds, axis_name='batch')
  merged_logits = jax.lax.all_gather(logits, axis_name='batch')
  merged_batch_mask = jax.lax.all_gather(batch_mask, axis_name='batch')

  aux = {
      'targets': merged_targets,
      'preds': merged_preds,
      'logits': merged_logits,
      'batch_mask': merged_batch_mask,
  }

  return metrics, aux


######################
# Loss Functions
######################
def focal_loss(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    *,
    label_weights: Optional[jnp.ndarray] = None,
    label_smoothing: Optional[float] = None,
    gamma: float = 2.0,
    pooling: Optional[str] = 'mean'
):
  """Focal loss (optionally averaged)."""
  loss = scenic_model_utils.focal_softmax_cross_entropy(
      logits=logits,
      one_hot_targets=one_hot_targets,
      weights=weights,
      label_weights=label_weights,
      label_smoothing=label_smoothing,
      gamma=gamma,
  )
  if pooling == 'mean':
    loss = jnp.mean(loss)

  return loss


def balanced_softmax_loss(
    logits: jnp.ndarray,
    one_hot_targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    samples_per_class: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Balanced Meta-Softmax Loss.

  As per: Balanced Meta-Softmax for Long-Tailed Visual Recognition.

  Args:
    logits: Output of model in shape [batch, ..., num_classes].
    one_hot_targets: One hot vector of shape [batch, ..., num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    samples_per_class: An array of shape [num_classses] housing the number of
      training samples per class.

  Returns:
    The mean BaLMS loss of the examples in the given batch as a scalar.
  """

  if samples_per_class is None:
    raise ValueError('samples_per_class cannot be None.')

  n_batch = logits.shape[0]
  spc = jnp.expand_dims(samples_per_class, axis=0)
  log_spc = jnp.log(spc)
  batch_log_spc = jnp.repeat(log_spc, n_batch, axis=0)
  logits_weighted = logits + batch_log_spc
  loss = scenic_model_utils.weighted_softmax_cross_entropy(
      logits=logits_weighted,
      one_hot_targets=one_hot_targets,
      weights=weights,
      label_smoothing=None,
      label_weights=None,
  )
  return loss


def get_classification_loss_fn(
    config: ml_collections.ConfigDict,
    dataset_meta_data: Dict[str, Any],
) -> LossFn:
  """Gets the classification loss function.

  Args:
    config: Configurations of the experiment.
    dataset_meta_data: Metadata of the dataset.

  Returns:
    A loss function.
  """

  # Ensure that targets are one hot encoded.
  assert dataset_meta_data['target_is_onehot']

  # Label weights
  if config.classification_loss.weighted_loss:
    label_weights = jnp.array(dataset_meta_data['label_weights'])
  else:
    label_weights = None

  # Get the loss function.
  loss_name = config.classification_loss.loss_name
  if loss_name == 'weighted_softmax_cross_entropy':
    label_smoothing = config.classification_loss.label_smoothing
    loss_fn = functools.partial(
        scenic_model_utils.weighted_softmax_cross_entropy,
        label_weights=label_weights,
        label_smoothing=label_smoothing,
    )
  elif loss_name == 'focal_softmax_cross_entropy':
    label_smoothing = config.classification_loss.label_smoothing
    focal_loss_gamma = config.classification_loss.focal_loss_gamma
    loss_fn = functools.partial(
        focal_loss,
        label_weights=label_weights,
        label_smoothing=label_smoothing,
        gamma=focal_loss_gamma
    )
  elif loss_name == 'balanced_softmax_loss':
    spc = jnp.array(dataset_meta_data['label_counts'])
    loss_fn = functools.partial(
        balanced_softmax_loss,
        samples_per_class=spc,
    )
  # TODO(girishvn): Add sigmoid cross entropy loss.
  # elif loss_name == 'sigmoid_cross_entropy':
  #   loss_fn = functools.partial(
  #       scenic_model_utils.sigmoid_cross_entropy,
  #       label_weights=label_weights,
  #   )
  else:
    raise ValueError(f'Unsupported loss name: {loss_name}')

  # Return.
  return loss_fn



