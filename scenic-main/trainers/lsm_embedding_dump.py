"""Trainer functions.

Adapted from google3/third_party/py/scenic/projects/multimask/trainer.py.

The below funcctions are adapted to allow for different input field names.
The original, expected field name was 'inputs', and it has here
been modified to 'input_signals'.
"""

import functools
import os
from typing import Dict, Type, Tuple, Union

from absl import logging
from clu import metric_writers
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np

from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
# To register the preprocessing ops
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_reconstruction_eval
from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_utils
from google3.pyglib import gfile


def pop_string_fields(batch: base_model.Batch):
  """Pop string fields from batch."""
  for k in list(batch.keys()):
    if 'str' in k:
      batch.pop(k)

  return batch


def dump_embeddings_metrics_fn(
    predictions: jnp.ndarray,
    prediction_masks: jnp.ndarray,
    batch: base_model.Batch,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, int]]:
  """Calculate metrics for the regression task."""

  del prediction_masks  # unused

  targets = batch['targets']
  batch_weights = batch.get('batch_mask')
  # create a mask with all data points, then chip at it based on input masks

  # get num examples
  num_examples = model_utils.num_examples(targets, predictions, batch_weights)
  psum_num_examples = jax.lax.psum(jnp.sum(num_examples), axis_name=axis_name)

  evaluated_metrics = {}
  evaluated_metrics['sample_count'] = (psum_num_examples, 1)  # normalizer is 1
  return evaluated_metrics


def dump_embeddings(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
):
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, valid_iter, test_iter, and
      meta_data.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_sate that has the state of training (including current global_step,
    model_state, rng, and the optimizer), train_summary and eval_summary which
    are dict of metrics (from the last evaluation and train metric logging
    respectively). These outputs are used for regression testing.
  """
  del workdir

  process_idx = jax.process_index()
  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  params_init_rng, dropout_init_rng = jax.random.split(rng, num=2)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}
  init_batch = next(dataset.valid_iter)
  (params, model_state, _, _) = (
      train_utils.initialize_model(
          model_def=model.flax_model,
          input_spec=[(
              init_batch['input_signal'].shape[1:],
              init_batch['input_signal'].dtype,
          )],
          config=config,
          rngs=init_rngs,
          train=False,  # so that masking and decoding in MAE are disabled.
      )
  )
  del init_batch  # free up memory for batch.

  # Create new / empty train_state.
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=None,
      tx=None,
      params=params,
      model_state=model_state,
      rng=None,
      metadata=None,
  )

  # If no checkpoint loaded from working directory, load from config.
  # BEGIN GOOGLE-INTERNAL
  # Get checkpoint path and config from XManager.
  if config.init_from.get('checkpoint_dir'):
    init_checkpoint_path = config.init_from.get('checkpoint_dir')
    restored_model_cfg = config
  elif config.init_from.get('xm'):
    xid, wid = config.init_from.get('xm')
    logging.info(f'Loading checkpoint from XID {xid} and WID {wid}')  # pylint: disable=logging-fstring-interpolation
    # Load parameters from the  previous train state.
    (restored_model_cfg, init_checkpoint_path) = (  # pylint: disable=unused-variable
        xm_utils.get_info_from_xmanager(xid, wid)
    )
  else:
    raise ValueError('No checkpoint XM info provided in config.')

  checkpoint_step = config.init_from.get('checkpoint_step', None)
  if checkpoint_step is None:
    raise ValueError('No checkpoint step provided in config.')
  # END GOOGLE-INTERNAL

  checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
  if checkpoint_format == 'scenic':
    restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
        init_checkpoint_path,
        train_state,
        assert_exist=True,
        step=checkpoint_step
    )
    # Load params from the restored checkpoint.
    train_state = lsm_mae_utils.restore_from_train_state(
        train_state, restored_train_state
    )
    del restored_train_state, restored_model_cfg
  else:
    raise ValueError(f'Unsupported checkpoint format: {checkpoint_format}')

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  ################### EVALUATION #######################
  train_state = train_utils.sync_model_state_across_replicas(train_state)

  # 2) Set up parallelized training and evaluation step functions.
  eval_step_pmapped = jax.pmap(
      functools.partial(
          lsm_mae_utils.eval_step,
          flax_model=model.flax_model,
          metrics_fn=dump_embeddings_metrics_fn,
          config=config,
          debug=config.debug_eval,
          train=False,
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )

  ################ EVALUATION ################
  # Set up dump directory.
  xm_id = config.init_from.checkpoint_dir.split('/')[-2]
  w_id = config.init_from.checkpoint_dir.split('/')[-1]
  experiment_name = f'{config.dataset_name}_embedding_dump_xid{xm_id}_wid{w_id}_{config.init_from.timestamp}'
  dump_dir = os.path.join(config.output_dir, experiment_name)
  gfile.MakeDirs(dump_dir)

  # Get items from config.
  batch_size = config.batch_size
  representation_layer = config.embedding_dump.representation_layer
  representation_aggregation_method = (
      config.embedding_dump.representation_aggregation_method
  )

  # Calculate steps per data split.
  # Iterate through data splits to dump.
  splits_to_dump = config.embedding_dump.splits_to_dump
  for split in splits_to_dump:

    # Set up task name: used for logging, file naming, and metric writing.
    task_name = f'embedding_dump_{split}'

    # Set up steps per split.
    if split == 'train':
      num_ex = dataset.meta_data['num_train_examples']
      ds_iter = dataset.train_iter
    elif split == 'valid':
      num_ex = dataset.meta_data['num_val_examples']
      ds_iter = dataset.valid_iter
    elif split == 'test':
      num_ex = dataset.meta_data['num_test_examples']
      ds_iter = dataset.test_iter
    else:
      raise ValueError(f'Unsupported split: {split}')

    # If the split does not exist continue.
    if ds_iter is None or num_ex is None:
      logging.info(f'Split {split} does not exist.')  # pylint: disable=logging-fstring-interpolation
      continue

    # Calculate total steps.
    total_steps = int(np.ceil(num_ex / batch_size))

    # Iterate through data splits to dump.
    eval_metrics = []
    for step in range(total_steps):

      data_batch = next(ds_iter)  # get next batch

      # Create dump batch.
      dump_batch = data_batch.copy()
      dump_batch.pop('input_signal')
      # Pop strings from data batch.
      data_batch = pop_string_fields(data_batch)
      e_metrics, _, eval_aux = eval_step_pmapped(  # pylint: disable=unused-variable
          train_state, data_batch
      )
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

      # Get embedding, add to dump batch, and dump to pickle.
      representation = eval_aux[representation_layer]  # get embeddings
      # Aggregate embeddings.
      if representation_aggregation_method == 'mean':
        representation = jnp.mean(representation, axis=-2)
      elif representation_aggregation_method == 'max':
        representation = jnp.max(representation, axis=-2)

      dump_batch[f'embedding_{representation_layer}'] = np.array(representation)
      lsm_mae_reconstruction_eval.save_dict_to_pickle(
          dump_batch,
          dump_dir,
          task_name + '_batch_' + str(step) + '_process_' + str(process_idx),
      )

    # Compute the sum over all examples in all step batches.
    eval_metrics = train_utils.stack_forest(eval_metrics)
    eval_metrics_summary = jax.tree_util.tree_map(
        lambda x: x.sum(), eval_metrics
    )
    del eval_metrics

    # write metrics to tensorboard
    # only write val[0] (number of examples) as normalization is not needed.
    writer.write_scalars(
        step=0,
        scalars={
            '_'.join((task_name, key)): val[0]
            for key, val in eval_metrics_summary.items()
        },
    )
    writer.flush()

    # Wait until computations are done before exiting.
    train_utils.barrier_across_hosts()
    logging.info(f'\n\nCompleted embedding dump for split {split}!\n\n')  # pylint: disable=logging-fstring-interpolation

  logging.info('\n\nCompleted embedding dump!\n\n')
  return 0
