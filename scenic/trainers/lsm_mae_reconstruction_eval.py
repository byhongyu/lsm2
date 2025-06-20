"""Comprehensive Downstream Evaluation Functions.

These functions run evaluation across all down stream task on the final trained
checkpoint of a given model. This can be run either after immedietly training or
on a model checkpoint.

Adapted from google3/third_party/py/scenic/projects/multimask/trainer.py.

The below funcctions are adapted to allow for different input field names.
The original, expected field name was 'inputs', and it has here
been modified to 'input_signals'.
"""

import copy
import datetime
import functools
import os
import pickle
import time  # pylint: disable=unused-import
from typing import Any, Dict, Optional, Type

from absl import logging
from clu import metric_writers
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.model_lib.base_models import base_model
# To register the preprocessing ops
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import train_utils

from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_utils
from google3.pyglib import gfile
from google3.experimental.largesensormodels.scenic.trainers.masking.masker_config import MaskStrategy_Config

from absl import flags

FLAGS = flags.FLAGS
from google3.experimental.largesensormodels.scenic.datasets import get_dataset


def reinitialize_validdataset(task_config, rng):
  task_config.dataset_configs.train_num_samples = 0
  task_config.dataset_configs.cache_dataset = False
  task_config.dataset_configs.shuffle_buffer_size = 0

  dataset = get_dataset.get_dataset(
      task_config, rng, dataset_service_address=FLAGS.dataset_service_address
  )
  return dataset


def save_dict_to_pickle(
    data_dict: Dict[str, Any], file_dir: str, filename_prefix: str
) -> None:
  """Saves a dictionary containing NumPy arrays to a pickle file, excluding keys with NoneType values.

  Args:
      data_dict: The dictionary to save.
      file_dir: The directory where the file should be saved.
      filename_prefix: The prefix for the filename (e.g., "my_data").
  """
  filtered_dict = {}
  for key, value in data_dict.items():
    if value is not None:
      filtered_dict[key] = value
  timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  filename = f'{filename_prefix}_{timestamp}.pickle'
  filepath = os.path.join(file_dir, filename)  # Combine directory and filename
  with gfile.Open(filepath, 'wb') as f:
    pickle.dump(filtered_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
  logging.info('Data saved to: %s', filepath)
  with gfile.Open(filepath, 'rb') as f:
    loaded_data = np.load(f, allow_pickle=True)
  for key in loaded_data.keys():
    logging.info(key)
    logging.info(len(loaded_data[key]))
    logging.info(loaded_data[key][0].shape)
    logging.info('-------------------')


def evaluate(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    task_name: str,
    model_cls: Type[base_model.BaseModel],
    restored_train_state: train_utils.TrainState,
    init_batch: lsm_mae_utils.Batch,
    dataset: dataset_utils.Dataset,
    writer: metric_writers.MetricWriter,
    step: int,
) -> Optional[Dict[str, Any]]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    task_name: Name of the sub-task (eg. forecast, imputation, etc.).
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    restored_train_state: The train state, passed from the trainer on which to
      run the evaluation.
    init_batch: The batch to use for initialization.
    dataset: The dataset that has train_iter, val_iter, test_iter, and
      meta_data.
    writer: CLU metrics writer instance.
    step: The current step of training.

  Returns:
    eval_summary which is a dict of metrics (from the last evaluation metric
    logging respectively). These outputs are used for regression testing.
  """

  ################ MODEL, LOSS SETUP ################
  # 1) Initialize model.
  model = model_cls(config, dataset.meta_data)
  rng, params_init_rng, dropout_init_rng = jax.random.split(rng, num=3)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}

  if config.masker_config.on_cpu:
    masked_tuple = (
        # this has had a bug here, but has never been thrown.
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

  (params, model_state, num_trainable_params, gflops) = (  # pylint: disable=unused-variable
      train_utils.initialize_model(
          model_def=model.flax_model,
          input_spec=[
              (
                  init_batch['input_signal'].shape[1:],
                  init_batch['input_signal'].dtype,
              ),
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

  del init_batch  # free up memory for batch.

  # 2) Create LR schedules and optimizer.
  schedule_fns = scenic_optax.make_schedule(config.get('schedule'))
  tx, _ = scenic_optax.make(config.optimizer, schedule_fns, params)
  opt_state = tx.init(params)

  # 3) Initialize the model train state with the optimizer state.
  rng, train_rng = jax.random.split(rng)  # pylint: disable=unused-variable
  train_state = train_utils.TrainState(
      global_step=0,
      opt_state=opt_state,
      tx=tx,
      params=params,
      model_state=model_state,
      rng=train_rng,
  )

  ################ RESTORE MODEL CHECKPOINT ################
  # Load params from the restored checkpoint.
  train_state = lsm_mae_utils.restore_from_train_state(
      train_state, restored_train_state
  )

  ################ PARALLELIZATION SETUP ################
  # 1) Set up replication across devices.
  # Replicate the optimzier, state, and rng across devices.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # 2) Set up parallelized training and evaluation step functions.
  eval_step_pmapped = jax.pmap(
      functools.partial(
          lsm_mae_utils.eval_step,
          flax_model=model.flax_model,
          metrics_fn=model.get_metrics_fn('validation'),
          config=config,
          debug=config.debug_eval,
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
  )

  ################ EVALUATION ################
  valid_iter = dataset.valid_iter
  num_valid_ex = dataset.meta_data['num_val_examples']

  eval_summary = {}
  if not isinstance(valid_iter, dict):  # Only on validation set.
    valid_iter, num_valid_ex = {'valid': valid_iter}, {'valid': num_valid_ex}
  dump_dir = None
  if config.get('enable_dump_mode', False):
    xm_id = config.init_from.checkpoint_dir.split('/')[-2]
    w_id = config.init_from.checkpoint_dir.split('/')[-1]
    experiment_name = f'{config.experiment_name}_{config.dataset_configs.valid_dataset}_xid_{xm_id}_wid_{w_id}_{config.init_from.timestamp}'
    dump_dir = os.path.join(config.output_dir, experiment_name)
    gfile.MakeDirs(dump_dir)
  for val_name, val_iter in valid_iter.items():
    num_ex = num_valid_ex[val_name]
    # Ceil rounding such that we include the last incomplete batch.
    eval_batch_size = config.get('eval_batch_size', config.batch_size)
    total_eval_steps = int(np.ceil(num_ex / eval_batch_size))
    steps_per_eval = config.get('steps_per_eval') or total_eval_steps
    eval_metrics = []
    for idx in range(steps_per_eval):  # pylint: disable=unused-variable
      logging.info('Running eval step %d of %d', idx, steps_per_eval)
      eval_batch = next(val_iter)
      dump_batch = {}
      if config.get('enable_dump_mode', False):
        dump_batch = eval_batch.copy()
        eval_batch.pop('key', None)
        eval_batch.pop('user_id', None)
      e_metrics, eval_plot_logits, eval_aux = eval_step_pmapped(  # pylint: disable=unused-variable
          train_state, eval_batch
      )
      if config.get('enable_dump_mode', False):
        dump_batch['eval_plot_logits'] = np.array(eval_plot_logits)
        dump_batch['token_mask'] = np.array(eval_aux['token_mask'])
        save_dict_to_pickle(
            dump_batch,
            dump_dir,
            task_name + '_step_' + str(idx),
        )
      eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

    # Update the eval summary.
    prefix = f'{task_name}/{val_name}'
    eval_summary.update(
        train_utils.log_eval_summary(
            step=step,
            eval_metrics=eval_metrics,
            writer=writer,
            prefix=prefix,
        )
    )
    del eval_metrics

  # Clean-up.
  writer.flush()
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train and eval summary after last step for regression testing.
  return eval_summary


# Random Imputation Evaluation
def random_imputation_eval(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    train_state: train_utils.TrainState,
    init_batch: lsm_mae_utils.Batch,
    dataset: dataset_utils.Dataset,
    writer: metric_writers.MetricWriter,
    step: int,
):
  """Evaluate the random imputation task."""

  random_imputation_config = config.random_imputation
  task_config = copy.deepcopy(config)

  eval_summary_dict = dict()
  for prob in random_imputation_config.random_imputation_ratios:

    task_name = f'random_imputation_{prob}_eval'  # Task name for logging

    if task_config.masker_config.inherited:
      # This also sets inheritance dependence to be True, such that the random
      # mask is randomly added ontop of a inherited mask until total
      # mask_probability is reached. This ensures that the same total masked
      # indices is the same.
      inherited_depend = True
    else:
      inherited_depend = None

    # update masker_config with the new maskstrategy with new probability that
    # we would like to evaluate with.
    task_config.masker_config.update_maskstrategy_list([
        MaskStrategy_Config(
            strategy='random',
            mask_probability=prob,
            inherited_depend=inherited_depend,
        )
    ])

    # force set to on_cpu to maintain consistency with other evals
    task_config.masker_config.on_cpu = True

    if task_config.masker_config.on_cpu:
      # need to re-intialize the validation dataset bc masking strat is done on cpu
      dataset = reinitialize_validdataset(task_config, rng)

    # Run reconstruction eval.
    eval_summary_dict[task_name] = evaluate(
        rng=rng,
        config=task_config,
        task_name=task_name,
        model_cls=model_cls,
        restored_train_state=train_state,
        init_batch=init_batch,
        dataset=dataset,
        writer=writer,
        step=step,
    )

  return eval_summary_dict


# Forecast Evaluation
def forecast_eval(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    train_state: train_utils.TrainState,
    init_batch: lsm_mae_utils.Batch,
    dataset: dataset_utils.Dataset,
    writer: metric_writers.MetricWriter,
    step: int,
):
  """Evaluate the forecast task."""
  forecast_config = config.forecast
  task_config = copy.deepcopy(config)

  eval_summary_dict = dict()
  for prob in forecast_config.horizons:
    # Update task name.
    task_name = f'forecast_{prob}_eval'  # Task name for logging
    # Update mask strategy.

    if task_config.masker_config.inherited:
      # inheritance dependence is set to false because we do not want the amt of
      # artifically introduced mask to be dependent on the inherited mask. if
      # we want to forecast 10 minutes, the bar size must encompass 10 min. It has
      # no effect if task_config.masker_config.inherited == False
      inherited_depend = False
    else:
      inherited_depend = None

    # update masker_config with the new maskstrategy with new probability that
    # we would like to evaluate with
    task_config.masker_config.update_maskstrategy_list([
        MaskStrategy_Config(
            strategy='bar',
            mask_probability=prob,
            mask_dim='time',
            mask_dim_contiguous=True,
            mask_dim_forecasting=True,
            inherited_depend=inherited_depend,
        )
    ])

    # TODO(xumax) This config format is only supported on_cpu because in order
    # to force forecasting, the correct config is MaskStrategy_Config(
    # strategy='forecasting', mask_probability=prob, mask_dim='time'), but this
    # config design is becoming deprecated in favor of above's config.
    task_config.masker_config.on_cpu = True

    if task_config.masker_config.on_cpu:
      # need to re-intialize the validation dataset bc masking strat is done on cpu
      dataset = reinitialize_validdataset(task_config, rng)

    # Run reconstruction eval.
    eval_summary_dict[task_name] = evaluate(
        rng=rng,
        config=task_config,
        task_name=task_name,
        model_cls=model_cls,
        restored_train_state=train_state,
        init_batch=init_batch,
        dataset=dataset,
        writer=writer,
        step=step,
    )

  return eval_summary_dict


# Imputation Evaluation
def imputation_eval(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    train_state: train_utils.TrainState,
    init_batch: lsm_mae_utils.Batch,
    dataset: dataset_utils.Dataset,
    writer: metric_writers.MetricWriter,
    step: int,
):
  """Evaluate the imputation task."""
  imputation_config = config.imputation
  task_config = copy.deepcopy(config)

  eval_summary_dict = dict()
  for prob in imputation_config.horizons:
    # Update task name.
    task_name = f'imputation_{prob}_eval'  # Task name for logging
    # Update mask strategy.

    if task_config.masker_config.inherited:
      # inheritance dependence is set to false because we do not want the amt of
      # artifically introduced mask to be dependent on the inherited mask. if
      # we want to impute 10 minutes, the bar size must encompass 10 min. It has
      # no effect if task_config.masker_config.inherited == False
      inherited_depend = False
    else:
      inherited_depend = None

    # update masker_config with the new maskstrategy with new probability that
    # we would like to evaluate with
    task_config.masker_config.update_maskstrategy_list([
        MaskStrategy_Config(
            strategy='bar',
            mask_probability=prob,
            mask_dim='time',
            mask_dim_contiguous=True,
            inherited_depend=inherited_depend,
        )
    ])

    # TODO(xumax) This config format is only supported on_cpu because in order
    # to force imputation, the correct config is MaskStrategy_Config(
    # strategy='imputation', mask_probability=prob, mask_dim='time'), but this
    # config design is becoming deprecated in favor of above's config.
    task_config.masker_config.on_cpu = True

    if task_config.masker_config.on_cpu:
      # need to re-intialize the validation dataset bc masking strat is done on cpu
      dataset = reinitialize_validdataset(task_config, rng)

    # Run reconstruction eval.
    eval_summary_dict[task_name] = evaluate(
        rng=rng,
        config=task_config,
        task_name=task_name,
        model_cls=model_cls,
        restored_train_state=train_state,
        init_batch=init_batch,
        dataset=dataset,
        writer=writer,
        step=step,
    )

  return eval_summary_dict


# # TODO(xumax, girishvn) fix later to work with new mask class configs. this is
# # commented out for now because documentaton is unclear on how it works with partialbar
# Sensor Imputation Evaluation
def sensor_imputation_eval(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Type[base_model.BaseModel],
    train_state: train_utils.TrainState,
    init_batch: lsm_mae_utils.Batch,
    dataset: dataset_utils.Dataset,
    writer: metric_writers.MetricWriter,
    step: int,
):
  raise ValueError(
      'does not work right now because legacy code used randompartialbar, which'
      ' was strange and not well documented'
  )


#   """Evaluate the imputation task."""
#   sensor_imputation_config = config.sensor_imputation
#   task_config = copy.deepcopy(config)
#   # this making strategy only works off cpu and not-inherited, therefore, we must turn it off manually
#   task_config.masker_config.on_cpu = False
#   task_config.masker_config.inherited = False

#   eval_summary_dict = dict()
#   for on_dim, off_dim in sensor_imputation_config.horizons:

#     # Update task name.
#     task_name = f'sensor_imputation_{on_dim}_{off_dim}_eval'
#     # Update mask strategy.
#     task_config.masked_feature_loss.token_mask_mask_probability = (
#         f'constant_1.0_partialbar_sensor_{on_dim}_{off_dim}'
#     )
#     task_config.masker_config.maskstrategy_list = [
#         MaskStrategy_Config(strategy='randompartialbar', mask_probability=prob)
#     ]
#     task_config.masker_config.maskstrategy_weights = np.array([1])

#     # Run reconstruction eval.
#     eval_summary_dict[task_name] = evaluate(
#         rng=rng,
#         config=task_config,
#         task_name=task_name,
#         model_cls=model_cls,
#         restored_train_state=train_state,
#         init_batch=init_batch,
#         dataset=dataset,
#         writer=writer,
#         step=step,
#     )

#   return eval_summary_dict
