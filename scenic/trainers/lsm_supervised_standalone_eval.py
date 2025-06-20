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
from scenic.projects.baselines import vit
# To register the preprocessing ops
from scenic.train_lib import optax as scenic_optax
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils

from google3.experimental.largesensormodels.scenic.trainers import lsm_supervised_utils
from google3.experimental.largesensormodels.scenic.utils import classification_utils as lsm_classification_utils


def init_from_train_state(
    train_state: Any,
    restored_train_state: Any,
):
  # Unfreeze parameters so they can be adjusted.
  params = flax.core.unfreeze(train_state.params)  # pylint: disable=unused-variable
  restored_params = flax.core.unfreeze(restored_train_state.params)
  return train_state.replace(params=flax.core.freeze(restored_params))


def supervised_evaluate(
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
  if model_cls is vit.ViTMultiLabelClassificationModel:
    model = model_cls(config, dataset.meta_data)
  else:
    model = model_cls(config.model, dataset.meta_data)

  # Initialize model.
  rng, params_init_rng, dropout_init_rng = jax.random.split(rng, num=3)
  init_rngs = {'params': params_init_rng, 'dropout': dropout_init_rng}
  init_batch = next(dataset.train_iter)

  # Define input specifications.
  input_spec = [
      (
          init_batch['input_signal'].shape[1:],
          init_batch['input_signal'].dtype,
      )
  ]
  if getattr(model, 'encode_metadata', False):
    input_spec.append(
        (
            init_batch['input_metadata'].shape[1:],
            init_batch['input_metadata'].dtype,
        )
    )

  # Initialize model.
  (params, model_state, _, _) = (
      train_utils.initialize_model(
          model_def=model.flax_model,
          input_spec=input_spec,
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

  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})  # pytype: disable=attribute-error

  # Load model checkpoint.
  # TODO(girishvn): adapt this to work with non-google code.
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
  init_step = config.init_from.checkpoint_step

  # Restore the checkpoint
  restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
      init_checkpoint_path, train_state=None, assert_exist=True, step=init_step
  )

  # Load params from the init_model.
  train_state = init_from_train_state(
      train_state, restored_train_state
  )
  del restored_train_state

  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Classification Labels - used for confusion matrix.
  label_names = dataset.meta_data['label_names']
  if label_names is None:
    label_names = [str(v) for v in dataset.meta_data['label_values']]

  # Setup parallel mapped train and eval step functions.
  eval_step_pmapped = jax.pmap(
      functools.partial(
          lsm_supervised_utils.eval_step,
          flax_model=model.flax_model,
          config=config,
          debug=config.debug_eval,
      ),
      axis_name='batch',
      # We can donate the eval_batch's buffer.
      donate_argnums=(1,),
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

  ################### EVALUATION #######################
  # Sync model state across replicas.
  train_state = train_utils.sync_model_state_across_replicas(train_state)
  eval_summary = evaluate(
      train_state,
      init_step,
      dataset.valid_iter,
      dataset.meta_data['num_val_examples'],
      dump_outputs=True,
  )

  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  logging.info('\n\nCompleted training and evaluation!\n\n')
  # Return the train and eval summary after last step for regression testing.
  return train_state, None, eval_summary
