"""Trainer functions.

Adapted from google3/third_party/py/scenic/projects/multimask/trainer.py.

The below funcctions are adapted to allow for different input field names.
The original, expected field name was 'inputs', and it has here
been modified to 'input_signals'.
"""

import functools
from typing import Any, Dict, Iterator, Optional, Tuple, Type

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
import flax
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.google.xm import xm_utils
from scenic.model_lib.base_models import base_model
# To register the preprocessing ops
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

from google3.experimental.largesensormodels.scenic.trainers import linear_probe_utils
from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_reconstruction_eval
from google3.experimental.largesensormodels.scenic.trainers import lsm_mae_utils


def train(
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
  model = model_cls(config, dataset.meta_data)

  # Initialize model.
  rng, params_init_rng, dropout_init_rng = jax.random.split(rng, num=3)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}
  init_batch = next(dataset.train_iter)

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

  (params, model_state, num_trainable_params, gflops) = (
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
  # NOTE: Do not delete init batch, as it may be used to init downstream models.

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

  # If a checkpoint exists in the working directory.
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state
    )
  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})  # pytype: disable=attribute-error

  if start_step == 0:
    logging.info(f'No checkpoint loaded from the working directory: {workdir}.')  # pylint: disable=logging-fstring-interpolation

  # If no checkpoint loaded from working directory, load from config.
  if (
      start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None
  ):
    logging.info('Loading checkpoint from config.')

    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    # BEGIN GOOGLE-INTERNAL
    # Get checkpoint path and config from XManager.
    if config.init_from.get('xm'):
      xid, wid = config.init_from.get('xm')
      logging.info(f'Loading checkpoint from XID {xid} and WID {wid}')  # pylint: disable=logging-fstring-interpolation
      (restored_model_cfg, init_checkpoint_path) = (
          xm_utils.get_info_from_xmanager(xid, wid)
      )
    # END GOOGLE-INTERNAL
    checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
    if init_checkpoint_path is not None:
      if checkpoint_format == 'scenic':
        restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
            init_checkpoint_path, train_state, assert_exist=True
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
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  # TODO(adosovitskiy): get rid of epochs?
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data
  )

  train_step_pmapped = jax.pmap(
      functools.partial(
          lsm_mae_utils.train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          lr_fns={name: lr_fn for _, name, (lr_fn, _) in schedule_fns},
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train,
      ),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )
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
  # TODO(xliucs) - Clean these comments.
  # if 'fewshot' in config:
  #   representation_fn_partial = functools.partial(
  #       representation_fn,
  #       flax_model=model.flax_model,
  #       representation_layer=config.fewshot.representation_layer,
  #   )

  #   fewshotter = fewshot_utils.FewShotEvaluator(
  #       representation_fn_partial, config.fewshot
  #   )

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

  # Evaluation function.
  def evaluate(
      train_state: train_utils.TrainState,
      step: int,
      valid_iter: Iterator[lsm_mae_utils.Batch],
      num_valid_ex: int,
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
      for _ in range(steps_per_eval):
        eval_batch = next(val_iter)
        e_metrics, eval_plot_logits, eval_aux = eval_step_pmapped(  # pylint: disable=unused-variable
            train_state, eval_batch
        )
        eval_metrics.append(train_utils.unreplicate_and_get(e_metrics))

      # Update the eval summary.
      eval_summary.update(
          train_utils.log_eval_summary(
              step=step,
              eval_metrics=eval_metrics,
              writer=writer,
              prefix=val_name,
          )
      )
      del eval_metrics
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
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      train_batch = next(dataset.train_iter)
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
      with report_progress.timed('eval'):
        eval_summary = evaluate(
            train_state,
            step,
            dataset.valid_iter,
            dataset.meta_data['num_val_examples'],
        )
      chrono.resume()

    ##################### LINEAR_PROBE EVALUATION ############################
    if 'linear_probe' in config:
      # Compute few-shot on-the-fly evaluation.
      representation_fn_linear_probe = functools.partial(
          lsm_mae_utils.representation_fn,
          flax_model=model.flax_model,
          representation_layer=config.linear_probe_representation_layer,
          gather_to_host=config.linear_probe_gather_to_host,
      )
      rng, linear_probe_rng = jax.random.split(rng)
      linearprober = linear_probe_utils.LinearEvaluator(
          representation_fn=representation_fn_linear_probe,
          rng=linear_probe_rng,
          linear_eval_config=config.linear_probe,
      )
      if (step % config.linear_probe_log_eval_steps == 1) or (
          step == total_steps
      ):
        with report_progress.timed('linear_probe'):
          results = linearprober.run_all(
              train_state,
              config.linear_probe,
              writer=writer,
              repr_step=step,
          )
          del results
          logging.info('Completed linear probe evaluation')
        writer.flush()

    #################### GENERATIVE EVALUATION ###########################
    # Imputation and Forecast Eval
    if 'forecast' in config:
      if (step % config.log_eval_steps == 1) or (step == total_steps):
        with report_progress.timed('forecast_eval'):
          lsm_mae_reconstruction_eval.forecast_eval(
              rng=rng,
              config=config,
              model_cls=model_cls,
              train_state=train_state,
              init_batch=init_batch,
              dataset=dataset,
              writer=writer,
              step=step,
          )
          logging.info('Completed forecast evaluation')
        writer.flush()

    if 'imputation' in config:
      if (step % config.log_eval_steps == 1) or (step == total_steps):
        with report_progress.timed('imputation_eval'):
          lsm_mae_reconstruction_eval.imputation_eval(
              rng=rng,
              config=config,
              model_cls=model_cls,
              train_state=train_state,
              init_batch=init_batch,
              dataset=dataset,
              writer=writer,
              step=step,
          )
          logging.info('Completed imputation evaluation')
        writer.flush()

    if 'random_imputation' in config:
      if (step % config.log_eval_steps == 1) or (step == total_steps):
        with report_progress.timed('randomimputation_eval'):
          lsm_mae_reconstruction_eval.random_imputation_eval(
              rng=rng,
              config=config,
              model_cls=model_cls,
              train_state=train_state,
              init_batch=init_batch,
              dataset=dataset,
              writer=writer,
              step=step,
          )
          logging.info('Completed random imputation evaluation')
        writer.flush()

    ##################### CHECKPOINTING ############################
    if (
        (step % checkpoint_steps == 1 and step > 1) or (step == total_steps)
    ) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
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
          )
          del unrep_train_state
      chrono.resume()  # Un-pause now.

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  logging.info('\n\nCompleted training and evaluation!\n\n')
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary
