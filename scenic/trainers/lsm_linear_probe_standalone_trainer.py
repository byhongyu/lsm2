"""Trainer functions.

Adapted from google3/third_party/py/scenic/projects/multimask/trainer.py.

The below funcctions are adapted to allow for different input field names.
The original, expected field name was 'inputs', and it has here
been modified to 'input_signals'.
"""

import functools
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Type, Union

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
import flax.linen as nn
import jax
from jax.nn import initializers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
from scenic.model_lib.base_models import base_model
from scenic.model_lib.layers import nn_layers
# To register the preprocessing ops
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_utils as lsm_model_utils
from google3.experimental.largesensormodels.scenic.trainers import lsm_supervised_utils
from google3.experimental.largesensormodels.scenic.utils import classification_utils as lsm_classification_utils


# Aliases for custom types:
Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


def reshape_time_crop_patch_embeddings(
    x,
    patch_reorder_shape: Tuple[int, int],
    start: Optional[float] = None,
    end: Optional[float] = None,
):
  """Reshape n_token embeddeding into an image of embeddedings."""
  # Get patch and input shape.
  n_h, n_w = patch_reorder_shape
  n_batch, n_tokens, embedding_dim = x.shape  # pylint: disable=unused-variable

  # Get start and end crop (along time axis).
  if end is None:
    end = 1
  if start is None:
    start = 0
  if start >= end:
    raise ValueError(f'start {start}, is greater than end {end}.')
  if start > 1 or end > 1:
    raise ValueError(f'start {start} and end {end} cannot be greater than 1.')

  # reorganize patches into image:
  x = jnp.reshape(x, [n_batch, n_h, n_w, embedding_dim])

  # Time Crop image based on horizon
  start_idx = int(start * n_h)
  end_idx = int(end * n_h)
  x = x[:, start_idx:end_idx, :, :]

  return x


class ConvBlocks(nn.Module):
  """Configurable number of convolutional blocks."""
  num_filters: Sequence[int]
  kernel_sizes: Sequence[int]
  use_bias: Optional[Union[bool, Sequence[bool]]]
  kernel_init: Initializer = jax.nn.initializers.lecun_normal()
  bias_init: Initializer = jax.nn.initializers.zeros
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    use_bias = True if self.use_bias is None else self.use_bias
    if not isinstance(use_bias, Iterable):
      use_bias = [use_bias] * len(self.num_filters)
    for n_filters, kernel_size, use_bias in zip(self.num_filters,
                                                self.kernel_sizes,
                                                use_bias):
      x = nn.Conv(
          features=n_filters,
          kernel_size=(kernel_size, kernel_size),
          strides=(1, 1),
          use_bias=use_bias,
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          dtype=self.dtype)(x)
      x = nn.relu(x)

    return x


class ConvPool(nn.Module):
  """Convolutional Pool Embeddedings.

  This takes a series of patch token embeddedings, and reorganizes them to
  reflect the original image shape. This takes the input of shape
  [batch, n_tokens, embedding_dim] and reshapes to
  [batch, n_h, n_w, embedding_dim], where n_h and n_w are the dimensions in the
  height and width dimension.

  patch_reorder_shape: The shape [n_h, n_w] to re-order the 'image'.
  start: A float [0, 1) percentage of the start time (along the h / time axis).
  end: A float (0, 1] percentage of the end time (along the h / time axis).
  num_filters: A sequence of ints, specifying how the number of filters per
    convolutional layer.
  kernel_sizes: A sequence of ints, specifying the kernel dimension per
    convolutional layer. Note. The kernel shape is assumed to be square.
  use_bias: A boolean sequence of whether or not to use a bias per layer.
  kernel_init: The initialization function for kernel weights.
  bias_init: The initialization function for bias weights.
  output_projection_type: Method ton project the output of the convolutional
    layers. Can be either 'flatten' (default) or 'mean'.
  dtype: The datatype to use for convolutional layers. Default is float32.

  """
  # Reshape params
  patch_reorder_shape: Tuple[int, int]
  start: float
  end: float
  # Conv block params
  num_filters: Sequence[int] = (20, 10)
  kernel_sizes: Sequence[int] = (3, 3)
  use_bias: Optional[Union[bool, Sequence[bool]]] = None
  kernel_init: Initializer = jax.nn.initializers.lecun_normal()
  bias_init: Initializer = jax.nn.initializers.zeros
  # Pooling
  output_projection_type: str = 'flatten'
  # Other
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    # Reshape
    self.reshape_fn = functools.partial(
        reshape_time_crop_patch_embeddings,
        patch_reorder_shape=self.patch_reorder_shape,
        start=self.start,
        end=self.end,
    )
    # Convolutional blocks
    self.conv_blocks = ConvBlocks(
        self.num_filters,
        self.kernel_sizes,
        self.use_bias,
        self.kernel_init,
        self.bias_init,
        self.dtype,
    )

  def __call__(self, x):
    x = self.reshape_fn(x)
    x = self.conv_blocks(x)

    if self.output_projection_type == 'mean':
      x = jnp.mean(x, axis=(1, 2))
    elif self.output_projection_type == 'flatten':
      x = jnp.reshape(x, [x.shape[0], -1])

    return x


class TimeWindowedMean(nn.Module):
  """Average of a time-window of embeddedings."""
  # Reshape params
  patch_reorder_shape: Tuple[int, int]
  start: float
  end: float

  def setup(self):
    # Reshape
    self.reshape_fn = functools.partial(
        reshape_time_crop_patch_embeddings,
        patch_reorder_shape=self.patch_reorder_shape,
        start=self.start,
        end=self.end,
    )

  def __call__(self, x):
    x = self.reshape_fn(x)
    x = jnp.mean(x, axis=(1, 2))
    return x


class LinearPool(nn.Module):
  """Linear Pooling."""
  aggregation_method: str = 'mean'
  output_size: int = 128
  use_bias: bool = True
  kernel_init: Initializer = jax.nn.initializers.lecun_normal()
  bias_init: Initializer = jax.nn.initializers.zeros
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  dtype: Optional[jnp.dtype] = jnp.float32

  def setup(self):
    # 1) Pooling.
    if self.aggregation_method == 'mean':
      self.pool_fn = functools.partial(jnp.mean, axis=1)
    elif self.aggregation_method == 'max':
      self.pool_fn = functools.partial(jnp.max, axis=1)
    else:
      raise ValueError(
          f'Unsupported aggregation_method: {self.aggregation_method}'
      )

    # 2) Projection.
    self.projection_layer = nn.Dense(
        features=self.output_size,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='output_projection',
    )

    # 3) Activation.
    if self.activation_fn is None:
      self.activation_fn = nn_layers.IdentityLayer

  def __call__(self, x):
    x = self.pool_fn(x)  # 1) Pooling.
    x = self.projection_layer(x)  # 2) Projection.
    x = self.activation_fn(x)  # 3) Activation.
    return x  # 3) Return.


class MaskedMeanPool(nn.Module):
  """Masked Mean Pooling.

  Given a matrix of inputs, of shape, [batch, n_items, item_dim], this
  function applies a mask along the items (axis=1) and the corresponding item
  values (dim=2), and calculates a mean of the non-masked items.

  TODO(girishvn, xumax): Brainstorm better naming for this, to better cohere
  with its usage (e.g. a attention mask aware mean pooling).
  """

  @nn.compact
  def __call__(self, x, mask):
    """Applies masked mean pooling to the embeddedings.

    Args:
      x: Inputs of shape [batch, n_items, item_dim].
      mask: The mask to apply to the embeddedings.
        Shape [batch, n_items, 1].

    Returns:
      The mean of pooled masked item values. Shape [batch, item_dim].
    """

    # mask of shape [batch, n_items, 1]
    mask_sum = jnp.sum(mask, axis=1)  # num unmasked items
    x_masked = x * mask
    x_masked_sum = jnp.sum(x_masked, axis=1)  # sum on item dim
    x_masked_mean = x_masked_sum / (mask_sum + 1e-6)  # normalize
    return x_masked_mean


class LinearProbingModel(nn.Module):
  """Linear head module for linear probing and finetuning."""

  num_classes: int
  embedding_model: nn.Module
  representation_layer: str
  representation_pooling_config: ml_collections.ConfigDict
  patch_reorder_shape: Optional[Tuple[int, int]] = None
  time_window: Optional[Tuple[float, float]] = None
  dropout_rate: Optional[float] = 0.5
  metadata_method: str = 'identity'

  def setup(self):
    # 1) Embedding pooling method. (Output fed to classification head).
    self.pool_method = self.representation_pooling_config.get('method', 'mean')

    # 1a) Mean pooling.
    if self.pool_method == 'mean':
      self.pool_fn = functools.partial(jnp.mean, axis=1)

    elif self.pool_method == 'attn_masked_mean':
      self.pool_fn = MaskedMeanPool()

    # 1b) Max pooling.
    elif self.pool_method == 'max':
      self.pool_fn = functools.partial(jnp.max, axis=1)
    # 1c) Trainable temporal convolutional pooling.
    # This embedding optionally crops the embeddings along the time dimension.
    elif self.pool_method == 'temporal_conv':
      # Get time window.
      time_window = self.representation_pooling_config.get('time_window', None)
      start, end = time_window if time_window is not None else (None, None)
      # Get patch reordering shape [n_h, n_w].
      patch_reorder_shape = self.representation_pooling_config.reorder_shape
      # Pool fn.
      self.pool_fn = ConvPool(
          patch_reorder_shape=patch_reorder_shape, start=start, end=end
      )
    # 1d) Mean pooling followed by projection layer.
    elif self.pool_method == 'learned_mean':
      self.pool_fn = LinearPool(aggregation_method='mean')
    # 1e) Max pooling followed by projection layer.
    elif self.pool_method == 'learned_max':
      self.pool_fn = LinearPool(aggregation_method='max')
    # 1f) Time windowed mean - averages a time window of embeddedings.
    elif self.pool_method == 'time_windowed_mean':
      # Get time window.
      time_window = self.representation_pooling_config.get('time_window', None)
      start, end = time_window if time_window is not None else (None, None)
      # Get patch reordering shape [n_h, n_w].
      patch_reorder_shape = self.representation_pooling_config.reorder_shape
      # Pool fn.
      self.pool_fn = TimeWindowedMean(
          patch_reorder_shape=patch_reorder_shape, start=start, end=end
      )
    # 1e) No operation pooling - don't do anything! 
    elif self.pool_method == 'noop':
      def noop(x):
        return x
      # Pool fn.
      self.pool_fn = noop
    # 1f) Unsupported pooling.
    else:
      self.pool_fn = None
      raise ValueError('Unsupported pooling function:', self.pool_method)

    # 2. Metadata embedding options.
    if self.metadata_method == 'identity':
      self.metadata_fn = nn_layers.IdentityLayer(name='metadata_fn')
    else:
      self.metadata_fn = None

    # 3. Dropout
    self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    # 4. Classification head.
    self.linear_layer = nn.Dense(
        self.num_classes,
        kernel_init=initializers.lecun_normal(),
        bias_init=initializers.zeros,
        name='output_projection',
    )

  # TODO(girishvn): add doc strings.
  def __call__(
      self,

      # Data inputs.
      x,
      x_metadata,

      # LSM ViT MAE specific inputs.
      mask_indices,
      unmasked_indices,
      token_mask,
      attn_mask,

      # Generic model inputs.
      train,
      finetune=False,
      debug=False,

      # Other inputs.
      **kwargs
  ):
    # 1) Get embedding model representation embedding.
    # TODO(girishvn): Support masking while LP training? Currently unsupported.
    _, aux = self.embedding_model(
        x,
        mask_indices=mask_indices,
        unmasked_indices=unmasked_indices,
        token_mask=token_mask,
        attn_mask=attn_mask,
        train=False
    )
    representation = aux[self.representation_layer]

    # 2) Add stop-gradient (if not full finetuning).
    representation = jax.lax.cond(
        jnp.logical_not(finetune),
        lambda x: jax.lax.stop_gradient(x),  # pylint: disable=unnecessary-lambda
        lambda x: x,
        representation,
    )

    # 3) Apply embedding pooling function.
    if isinstance(self.pool_fn, MaskedMeanPool):
      # Pools embeddings, passed from the encoder, using the encoder attention
      # mask (attn_mask). The mean is normalized by the sum of the mask, in this
      # case the number of unmasked tokens.
      # attn_mask is originally of shape [batch, 1, num_tokens], but must be
      # [batch, num_tokens, 1] to broken cast to the representation shape.
      attn_mask = jnp.transpose(attn_mask, axes=(0, 2, 1))
      pooled_representation = self.pool_fn(representation, attn_mask)
    else:
      pooled_representation = self.pool_fn(representation)

    # 4) Dropout
    pooled_representation = self.dropout_layer(
        pooled_representation, deterministic=not train
    )

    # 5) Add metadata embedding.
    if self.metadata_fn is not None and x_metadata is not None:
      metadata_representation = self.metadata_fn(x_metadata)
      pooled_representation = jnp.concatenate(
          [pooled_representation, metadata_representation], axis=-1
      )

    # 6) Pass pooled embeddeding to classsification head.
    linear_output = self.linear_layer(pooled_representation)
    # 7) Return linear output and aux.
    return linear_output, aux

  def init_embeddeding_from_train_state(
      self,
      train_state: Any,
      restored_train_state: Any,
      model_key: str = 'embedding_model',
  ):
    # Unfreeze parameters so they can be adjusted.
    params = flax.core.unfreeze(train_state.params)
    restored_params = flax.core.unfreeze(restored_train_state.params)

    # Copy over parameters from restored train state.
    for m_key, m_params in restored_params.items():
      if m_key in params[model_key].keys():

        jax.debug.print(f'Copying over params: {m_key}')
        params[model_key][m_key] = m_params

    # Replace the parameters of train state and return.
    return train_state.replace(params=flax.core.freeze(params))


def linear_probe_train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Optional[Dict[str, float]], Optional[Dict[str, Any]]]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, val_iter, test_iter, and
      meta_data.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_sate that has the state of training (including current global_step,
    model_state, rng, and the optimizer), train_summary and eval_summary which
    are dict of metrics (from the last evaluation and train metric logging
    respectively). These outputs are used for regression testing.
  """
  lead_host = jax.process_index() == 0
  # Build the loss_fn, metrics, and flax_model.
  backbone_model = model_cls(config, dataset.meta_data)

  # Calculate the number of patches / patch shape per input
  n_h, n_w = lsm_model_utils.calculate_patched_img_shape(dataset, config)
  config.representation_pooling.update({'reorder_shape': (n_h, n_w)})
  # Get model definition.
  lp_model = LinearProbingModel(
      num_classes=dataset.meta_data['num_classes'],
      embedding_model=backbone_model.flax_model,
      representation_layer=config.linear_probe_representation_layer,
      representation_pooling_config=config.representation_pooling,
      dropout_rate=config.linear_dropout_rate,
      metadata_method=config.linear_probe_metadata_method,
  )

  # Initialize model.
  rng, params_init_rng, dropout_init_rng = jax.random.split(rng, num=3)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}
  init_batch = next(dataset.train_iter)

  # Define aspects of the input spec.
  # Require the flexible get operation (though prone to silent errors) as not
  # all LP configs require a 'masking' config.
  cpu_masking = config.masker_config.get('on_cpu', False)
  inherited_masking = config.masker_config.get('inherited', False)
  if cpu_masking and inherited_masking:
    masked_tuple = (
        init_batch['mask_indices'].shape[1:],
        init_batch['mask_indices'].dtype,
    )
    unmasked_tuple = (
        init_batch['unmasked_indices'].shape[1:],
        init_batch['unmasked_indices'].dtype,
    )
    token_mask_tuple = (
        init_batch['token_mask'].shape[1:],
        init_batch['token_mask'].dtype,
    )
    if init_batch['attn_mask'] is not None:
      attn_mask_tuple = (
          init_batch['attn_mask'].shape[1:],
          init_batch['attn_mask'].dtype,
      )
    else:
      attn_mask_tuple = None
  else:
    unmasked_tuple = None
    masked_tuple = None
    token_mask_tuple = None
    attn_mask_tuple = None

  # Check Metadata features.
  if init_batch['input_metadata'] is not None:
    metadata_tuple = (
        init_batch['input_metadata'].shape[1:],
        init_batch['input_metadata'].dtype,
    )
  else:
    metadata_tuple = None
    if (
        config.linear_probe_metadata_method is not None and
        config.linear_probe_metadata_method != 'none'
    ):
      raise ValueError(
          'Metadata method is specified but metadata is not available in batch.'
      )

  # Initialize compute graph.
  (params, model_state, num_trainable_params, gflops) = (
      train_utils.initialize_model(
          model_def=lp_model,
          input_spec=[
              (
                  init_batch['input_signal'].shape[1:],
                  init_batch['input_signal'].dtype,
              ),
              metadata_tuple,
              masked_tuple,
              unmasked_tuple,
              token_mask_tuple,
              attn_mask_tuple,
          ],
          config=config,
          rngs=init_rngs,
          train=True,  # so that masking and decoding in MAE are initialized
      )
  )
  del init_batch

  # Create LR schedules and optimizer.
  schedule_fns = scenic_optax.make_schedule(config.get('schedule'))
  tx, _ = scenic_optax.make(config.optimizer, schedule_fns, params)
  opt_state = tx.init(params)

  rng, train_rng = jax.random.split(rng)  # pylint: disable=unused-variable

  # Create chrono class to track and store training statistics and metadata:
  chrono = train_utils.Chrono()

  # Create new / empty train_state.
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
      metadata={'chrono': chrono.save()},
  )
  start_step = train_state.global_step

  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})  # pytype: disable=attribute-error

  # Load model checkpoint.
  # TODO(girishvn): adapt this to work with non-google code.
  # TODO(girishvn): add support for fault tolerant training. E.g. loading model
  # from latest working dir checkpoint.
  # BEGIN GOOGLE-INTERNAL
  # Get checkpoint path and config from XManager.
  if config.init_from.get('checkpoint_dir'):
    init_checkpoint_path = config.init_from.get('checkpoint_dir')
  elif config.init_from.get('xm'):
    xid, wid = config.init_from.get('xm')
    # Load parameters from the  previous train state.
    (restored_model_cfg, init_checkpoint_path) = (  # pylint: disable=unused-variable
        xm_utils.get_info_from_xmanager(xid, wid)
    )
  else:
    raise ValueError('No checkpoint XM info provided in config.')
  # END GOOGLE-INTERNAL
  # Get checkpoint step.
  init_step = config.init_from.get('checkpoint_step', None)

  # Restore the checkpoint
  restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
      init_checkpoint_path, train_state=None, assert_exist=True, step=init_step
  )

  # Load params from the init_model.
  train_state = lp_model.init_embeddeding_from_train_state(
      train_state, restored_train_state
  )
  del restored_train_state

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data
  )

  # Classification Labels - used for confusion matrix.
  label_names = dataset.meta_data['label_names']
  if label_names is None:
    label_names = [str(v) for v in dataset.meta_data['label_value']]

  # Label weights
  loss_fn = lsm_supervised_utils.get_classification_loss_fn(
      config, dataset.meta_data
  )

  # Setup parallel mapped train and eval step functions.
  train_step_pmapped = jax.pmap(
      functools.partial(
          lsm_supervised_utils.train_step,
          flax_model=lp_model,
          lr_fns={name: lr_fn for _, name, (lr_fn, _) in schedule_fns},
          loss_fn=loss_fn,
          max_grad_norm=config.get('max_grad_norm', None),
          config=config,
          debug=config.debug_train,
          has_aux=True,
          finetune=config.linear_finetune,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
  eval_step_pmapped = jax.pmap(
      functools.partial(
          lsm_supervised_utils.eval_step,
          flax_model=lp_model,
          config=config,
          debug=config.debug_eval,
          has_aux=True,
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )

  # Get logging, checkpointing and model syncing steps.
  log_eval_steps = config.get('log_eval_steps')
  if not log_eval_steps:
    raise ValueError("'log_eval_steps' should be specified in the config.")
  checkpoint_steps = config.get('checkpoint_steps') or log_eval_steps
  log_summary_steps = config.get('log_summary_steps') or log_eval_steps
  model_sync_steps = config.get('model_sync_steps') or 1

  # Ensure that checkpoint and eval are frequency aligned with model syncing.
  if checkpoint_steps % model_sync_steps != 0:
    raise ValueError(
        f'checkpoint_steps {checkpoint_steps} should be a multiple '
        f'of model_sync_steps {model_sync_steps}.'
    )
  if log_eval_steps % model_sync_steps != 0:
    raise ValueError(
        f'log_eval_steps {log_eval_steps} should be a multiple '
        f'of model_sync_steps {model_sync_steps}.'
    )

  # Setup evaluation routine.
  def evaluate(
      train_state: train_utils.TrainState,
      step: int,
      valid_iter: Iterator[lsm_supervised_utils.Batch],
      num_valid_ex: int,
      dump_outputs: bool = False,
  ) -> Dict[str, Any]:
    eval_summary = {}
    if not isinstance(valid_iter, dict):  # Only on validation set.
      valid_iter, num_valid_ex = {'valid': valid_iter}, {'valid': num_valid_ex}

    for val_name, val_iter in valid_iter.items():
      num_ex = num_valid_ex[val_name]
      # Ceil rounding such that we include the last incomplete batch.
      eval_batch_size = config.get('eval_batch_size', config.batch_size)
      total_eval_steps = int(np.ceil(num_ex / eval_batch_size))
      steps_per_eval = config.get('steps_per_eval') or total_eval_steps
      eval_metrics = []
      eval_targets = []
      eval_preds = []
      eval_logits = []
      for _ in range(steps_per_eval):
        eval_batch = next(val_iter)
        e_metrics, e_aux = eval_step_pmapped(
            train_state, eval_batch
        )
        eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))
        # Pass out more granular metrics.
        e_aux = train_utils.unreplicate_and_get(e_aux)
        targets = jnp.ravel(e_aux['targets'])
        preds = jnp.ravel(e_aux['preds'])
        logits = jnp.reshape(e_aux['logits'], (-1, e_aux['logits'].shape[-1]))
        batch_mask = jnp.ravel(e_aux['batch_mask'])
        mask_idx = batch_mask == 1
        eval_targets += targets[mask_idx].tolist()
        eval_preds += preds[mask_idx].tolist()
        eval_logits.append(logits[mask_idx])

      # Update the eval summary.
      eval_summary.update(
          train_utils.log_eval_summary(
              step=step,
              eval_metrics=eval_metrics,
              writer=writer,
              prefix=val_name,
          )
      )
      # Additional classification metrics.
      eval_logits = jnp.concatenate(eval_logits, axis=0)
      lsm_classification_utils.classification_metrics(
          targets=eval_targets,
          preds=eval_preds,
          logits=eval_logits,
          label_names=label_names,
          step=step,
          writer=writer,
          write_out_files=lead_host,
          workdir=workdir,
          prefix=val_name,
      )

      # Dump outputs (targets, preds, logits) to file.
      if dump_outputs:
        lsm_classification_utils.dump_classification_outputs(
            targets=eval_targets,
            preds=eval_preds,
            logits=eval_logits,
            label_names=label_names,
            step=step,
            writer=writer,
            write_out_files=lead_host,
            workdir=workdir,
            prefix=val_name,
        )

      # Clean up.
      del eval_metrics, eval_targets, eval_preds

    writer.flush()
    return eval_summary

  train_metrics, extra_training_logs = [], []
  train_summary, eval_summary = None, None

  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  logging.info('Starting training loop at step %d.', start_step + 1)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=total_steps,
      writer=writer,
      every_secs=None,
      every_steps=config.get('report_progress_step', log_summary_steps),
  )

  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)

  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))

  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)

  flax.config.update('flax_use_orbax_checkpointing', False)
  write_note(f'First step compilations...\n{chrono.note}')

  # Iterate over training steps.
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      # Get data batch.
      train_batch = next(dataset.train_iter)
      # Apply train step.
      train_state, t_metrics, t_logs = train_step_pmapped(
          train_state, train_batch
      )
      # This will accumulate metrics in TPU memory up to the point that we log
      # them. This is no problem for small metrics but may be a problem for
      # large (e.g. segmentation) metrics. An alternative is to set
      # `log_summary_steps` to a small number, or to use
      # `train_utils.unreplicate_and_get` here instead of right before writing
      # summaries, but that means in each step, we have data transfer between
      # tpu and host, which might slow down the training.
      train_metrics.append(t_metrics)
      # Additional training logs: learning rate:
      t_logs = jax.tree_util.tree_map(jax_utils.unreplicate, t_logs)
      extra_training_logs.append(t_logs)

    # Quick indication that training is happening.
    logging.log_first_n(logging.INFO, 'Finished training step %d.', 5, step)
    for h in hooks:
      h(step)

    # Sync model state across replicas.
    # a. If model_sync_steps is 1, sync every step.
    # b. If model_sync_steps is not 1, sync every model_sync_steps.
    # c. Sync at the end of training.
    # NOTE: Using (step % model_sync_steps == 1) sync frequency to cohere with
    # scenic logging and eval frequency.
    if (
        model_sync_steps == 1 or
        step % model_sync_steps == 1 or
        step == total_steps
    ):
      train_state = train_utils.sync_model_state_across_replicas(train_state)

    ############### LOG TRAIN SUMMARY ###############
    if (
        (step % log_summary_steps == 1)
        or (step == total_steps)
        or (lead_host and chrono.warmup)
    ):
      chrono.pause(wait_for=(train_metrics))
      logging.info('Logging train summary at step %d.', step)
      if lead_host:
        chrono.tick(step, writer, write_note)
      # train_metrics is list of a dictionaries of metrics, where the shape of
      # the metrics[key] is [n_local_devices]. However, because metric functions
      # have a psum, we have already summed across the whole sharded batch, and
      # what's returned is n_local_devices copies of the same summed metric.
      # So we do unreplicate and fetch them to host using `unreplicate_and_get`.
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(
              train_utils.unreplicate_and_get, train_metrics
          ),
          extra_training_logs=jax.tree_util.tree_map(
              jax.device_get, extra_training_logs
          ),
          writer=writer,
      )
      # Reset metric accumulation for next evaluation cycle.
      train_metrics, extra_training_logs = [], []
      chrono.resume()

    ################### EVALUATION #######################
    if (step % log_eval_steps == 1) or (step == total_steps):
      chrono.pause(wait_for=(train_state.params))
      logging.info('Running eval at step %d.', step)
      with report_progress.timed('eval'):
        dump_outputs = (step == total_steps)
        eval_summary = evaluate(
            train_state=train_state,
            step=step,
            valid_iter=dataset.valid_iter,
            num_valid_ex=dataset.meta_data['num_val_examples'],
            dump_outputs=dump_outputs,
        )
      chrono.resume()

    ##################### CHECKPOINTING ############################
    if (
        (step % checkpoint_steps == 1 and step > 1) or (step == total_steps)
    ) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      logging.info('Saving checkpoint at step %d.', step)
      with report_progress.timed('checkpoint'):
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          train_utils.save_checkpoint(
              workdir,
              unrep_train_state,
              max_to_keep=config.max_checkpoints_to_keep,
              overwrite=True,
          )
          del unrep_train_state
      chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  logging.info('\n\nCompleted training and evaluation!\n\n')
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary



