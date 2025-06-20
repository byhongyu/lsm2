# TODO(xumax, xliucs) someone please merge this with the lsm_mae_reconstruction_eval.py

"""Trainer functions.

Adapted from google3/third_party/py/scenic/projects/multimask/trainer.py.

The below funcctions are adapted to allow for different input field names.
The original, expected field name was 'inputs', and it has here
been modified to 'input_signals'.
"""

from typing import Type

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
from scenic.model_lib.base_models import base_model
# To register the preprocessing ops
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_reconstruction_eval
from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_utils


def evaluate(
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

  # Build the loss_fn, metrics, and flax_model.
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  rng, params_init_rng, dropout_init_rng = jax.random.split(rng, num=3)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}
  init_batch = next(dataset.valid_iter)
  if config.masker_config.on_cpu:
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

  restored_model_cfg = config.init_from.get('model_config')
  init_checkpoint_path = config.init_from.get('checkpoint_path')
  checkpoint_step = config.init_from.get('checkpoint_step', None)
  if checkpoint_step is None:
    raise ValueError('No checkpoint step provided in config.')

  # Get checkpoint path and config from XManager.
  if config.init_from.get('xm'):
    xid, wid = config.init_from.get('xm')
    logging.info(f'Loading checkpoint from XID {xid} and WID {wid}')  # pylint: disable=logging-fstring-interpolation
    (restored_model_cfg, init_checkpoint_path) = (
        xm_utils.get_info_from_xmanager(xid, wid)
    )
  elif config.init_from.get('checkpoint_dir'):
    init_checkpoint_path = config.init_from.get('checkpoint_dir')
    restored_model_cfg = config
  else:
    raise ValueError('No checkpoint XM info provided in config.')

  checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
  if init_checkpoint_path is not None:
    if checkpoint_format == 'scenic':
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path,
          train_state,
          assert_exist=True,
          step=checkpoint_step,
      )
      # Load params from the restored checkpoint.
      train_state = lsm_mae_utils.restore_from_train_state(
          train_state, restored_train_state
      )
      del restored_train_state, restored_model_cfg
    else:
      raise ValueError(f'Unsupported checkpoint format: {checkpoint_format}')
  else:
    logging.info('No checkpoint path specified in the config.')

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  train_state = train_utils.sync_model_state_across_replicas(train_state)
  del params  # Do not keep a copy of the initial params.

  ################### EVALUATION #######################
  # 1 RECONSTRUCTION EVALUATION ###########################
  # Imputation and Forecast Eval
  if 'random_imputation' in config:
    lsm_mae_reconstruction_eval.random_imputation_eval(
        rng=rng,
        config=config,
        model_cls=model_cls,
        train_state=train_state,
        init_batch=init_batch,
        dataset=dataset,
        writer=writer,
        step=checkpoint_step,
    )
    logging.info('Completed random evaluation')
    writer.flush()
  if 'forecast' in config:
    lsm_mae_reconstruction_eval.forecast_eval(
        rng=rng,
        config=config,
        model_cls=model_cls,
        train_state=train_state,
        init_batch=init_batch,
        dataset=dataset,
        writer=writer,
        step=checkpoint_step,
    )
    logging.info('Completed forecast evaluation')
    writer.flush()

  if 'imputation' in config:
    lsm_mae_reconstruction_eval.imputation_eval(
        rng=rng,
        config=config,
        model_cls=model_cls,
        train_state=train_state,
        init_batch=init_batch,
        dataset=dataset,
        writer=writer,
        step=checkpoint_step,
    )
    logging.info('Completed imputation evaluation')
    writer.flush()

  if 'sensor_imputation' in config:
    lsm_mae_reconstruction_eval.sensor_imputation_eval(
        rng=rng,
        config=config,
        model_cls=model_cls,
        train_state=train_state,
        init_batch=init_batch,
        dataset=dataset,
        writer=writer,
        step=checkpoint_step,
    )
    logging.info('Completed sensor imputation evaluation')
    writer.flush()

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  logging.info('\n\nCompleted training and evaluation!\n\n')
  # Return the train and eval summary after last step for regression testing.
  return
