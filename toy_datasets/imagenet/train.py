"""Methods for training ResNet-50 on ImageNet using JAX."""

from collections.abc import Callable, Mapping, Sequence
import functools
from typing import Any

from absl import logging
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
from clu import platform
from etils import epath
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
import grain.tensorflow as grain
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from orbax import checkpoint as orbax_checkpoint

from google3.experimental.largesensormodels.toy_datasets.imagenet import input_pipeline
from google3.experimental.largesensormodels.toy_datasets.imagenet import resnet_v1


@flax.struct.dataclass
class TrainState:
  """State of the model and the training.

  This includes parameters, statistics and optimizer.
  """

  step: int
  params: Any
  opt_state: optax.OptState
  batch_stats: Any


def _get_checkpoint_manager(
    workdir: epath.PathLike,
) -> orbax_checkpoint.CheckpointManager:
  # The keys in this dict should match the keys in `checkpointed_state`.
  checkpointers = dict(
      train_state=orbax_checkpoint.PyTreeCheckpointer(),
      train_iter=orbax_checkpoint.Checkpointer(grain.OrbaxCheckpointHandler()),  # pytype:disable=wrong-arg-types
  )
  checkpoint_dir = epath.Path(workdir) / "checkpoints"
  return orbax_checkpoint.CheckpointManager(
      checkpoint_dir,
      checkpointers=checkpointers,
      options=orbax_checkpoint.CheckpointManagerOptions(create=True),
  )


def load_last_state(
    workdir: epath.PathLike, checkpointed_state: Mapping[str, Any] | None
) -> TrainState:
  """Loads the last state from Orbax.

  Args:
    workdir: The working directory to store Orbax state.
    checkpointed_state: an optional dictionary of object name ("train_state" and
      "train_iter") to restorable object. `None` to let Orbax restore from disk.

  Returns:
    The last checkpointed train state.
  """
  checkpoint_manager = _get_checkpoint_manager(workdir)
  if checkpoint_manager.latest_step() is None:
    raise ValueError("No last step found. Orbax has not run yet.")
  checkpointed_state = checkpoint_manager.restore(
      checkpoint_manager.latest_step(), items=checkpointed_state
  )
  return checkpointed_state["train_state"]


def merge_batch_stats(replicated_state: TrainState) -> TrainState:
  """Merge model batch stats."""
  if jax.tree.leaves(replicated_state.batch_stats):
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "x"), "x")
    return replicated_state.replace(
        batch_stats=cross_replica_mean(replicated_state.batch_stats)
    )
  else:
    return replicated_state


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: np.ndarray,
    input_shape: Sequence[int],
    num_classes: int,
    learning_rate_fn: Callable[[int], float],
) -> tuple[nn.Module, optax.GradientTransformation, TrainState]:
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    input_shape: Shape of the inputs fed into the model.
    num_classes: Number of classes in the output layer.
    learning_rate_fn: Function returning learning rate given step number.

  Returns:
    The model, optimizer, and initial train state.
  """
  if config.model_name == "resnet18":
    model_cls = resnet_v1.ResNet18
  elif config.model_name == "resnet50":
    model_cls = resnet_v1.ResNet50
  else:
    raise ValueError(f"Model {config.model_name} not supported.")
  model = model_cls(num_classes=num_classes)
  variables = model.init(rng, jnp.ones(input_shape), train=False)  # pytype: disable=wrong-arg-types
  params = variables["params"]
  batch_stats = variables["batch_stats"]
  parameter_overview.log_parameter_overview(params)
  optimizer = optax.sgd(  # pytype: disable=wrong-arg-types  # numpy-scalars
      learning_rate=learning_rate_fn, momentum=config.sgd_momentum
  )
  return (
      model,
      optimizer,
      TrainState(
          step=0,
          params=params,
          opt_state=optimizer.init(params),
          batch_stats=batch_stats,
      ),
  )


def cross_entropy_loss(*, logits, labels):
  logp = jax.nn.log_softmax(logits)
  loglik = jnp.take_along_axis(logp, labels[:, None], axis=1)
  return -loglik


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
  accuracy: metrics.Accuracy
  eval_loss: metrics.Average.from_output("loss")


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
  train_accuracy: metrics.Accuracy
  learning_rate: metrics.LastValue.from_output("learning_rate")
  train_loss: metrics.Average.from_output("loss")
  train_loss_std: metrics.Std.from_output("loss")


def cosine_decay(lr: float, current_epoch: float, total_epochs: float) -> float:
  ratio = jnp.maximum(0.0, current_epoch / total_epochs)
  mult = 0.5 * (1.0 + jnp.cos(jnp.pi * ratio))
  return mult * lr  # pytype: disable=bad-return-type  # jax-types


def get_learning_rate(
    step: int,
    *,
    base_learning_rate: float,
    steps_per_epoch: int,
    num_epochs: float,
    warmup_epochs: float = 5.0,
) -> float:
  """Cosine learning rate schedule."""
  logging.info(
      "get_learning_rate(step=%s, base_learning_rate=%s, steps_per_epoch=%s,"
      " num_epochs=%s",
      step,
      base_learning_rate,
      steps_per_epoch,
      num_epochs,
  )
  if steps_per_epoch <= 0:
    raise ValueError(
        "steps_per_epoch should be a positive integer but was "
        f"{steps_per_epoch}."
    )
  if warmup_epochs >= num_epochs:
    raise ValueError(
        "warmup_epochs should be smaller than num_epochs. "
        f"Currently warmup_epochs is {warmup_epochs}, "
        f"and num_epochs is {num_epochs}."
    )
  epoch = step / steps_per_epoch
  lr = cosine_decay(
      base_learning_rate, epoch - warmup_epochs, num_epochs - warmup_epochs
  )
  warmup = jnp.minimum(1.0, epoch / warmup_epochs)
  return lr * warmup  # pytype: disable=bad-return-type  # jax-types


def train_step(
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    train_state: TrainState,
    batch: Mapping[str, jnp.ndarray],
    learning_rate_fn: Callable[[int], float],
    weight_decay: float,
) -> tuple[TrainState, metrics.Collection]:
  """Perform a single training step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    optimizer: Optax optimizer.
    train_state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    learning_rate_fn: Function returning learning rate given step number.
    weight_decay: Weighs L2 regularization term.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  def loss_fn(params):
    variables = {"params": params, "batch_stats": train_state.batch_stats}
    logits, new_variables = model.apply(
        variables, batch["image"], mutable=["batch_stats"], train=True
    )
    loss = jnp.mean(cross_entropy_loss(logits=logits, labels=batch["label"]))
    weight_penalty_params = jax.tree.leaves(variables["params"])
    weight_l2 = sum(
        [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1]
    )
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_variables["batch_stats"], logits)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_batch_stats, logits)), grad = grad_fn(train_state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")
  updates, new_opt_state = optimizer.update(
      grad, train_state.opt_state, train_state.params
  )
  new_params = optax.apply_updates(train_state.params, updates)
  new_state = train_state.replace(  # pytype: disable=attribute-error
      step=train_state.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      batch_stats=new_batch_stats,
  )

  metrics_update = TrainMetrics.gather_from_model_output(
      loss=loss,
      logits=logits,
      labels=batch["label"],
      learning_rate=learning_rate_fn(train_state.step),
  )
  return new_state, metrics_update


@functools.partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=0)
def eval_step(
    model: nn.Module, train_state: TrainState, batch: Mapping[str, jnp.ndarray]
) -> metrics.Collection:
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    train_state: Replicate model state.
    batch: Inputs that should be evaluated.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step(batch=%s)", batch)
  variables = {
      "params": train_state.params,
      "batch_stats": train_state.batch_stats,
  }
  logits = model.apply(variables, batch["image"], mutable=False, train=False)
  loss = jnp.mean(cross_entropy_loss(logits=logits, labels=batch["label"]))
  return EvalMetrics.gather_from_model_output(
      logits=logits,
      labels=batch["label"],
      loss=loss,
      mask=batch.get("mask"),
  )


class StepTraceContextHelper:
  """Helper class to use jax.profiler.StepTraceAnnotation as recommended in go/xprof-instrument-jax."""

  def __init__(self, name: str, init_step_num: int):
    self.name = name
    self.step_num = init_step_num
    self.context = None

  def __enter__(self):
    self.context = jax.profiler.StepTraceAnnotation(
        self.name, step_num=self.step_num
    )
    self.step_num += 1
    self.context.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, tb):
    assert self.context is not None, "Exited context without entering."
    self.context.__exit__(exc_type, exc_value, tb)
    self.context = None

  def next_step(self):
    if self.context is None:
      raise ValueError("Must call next_step() within a context.")
    self.__exit__(None, None, None)
    self.__enter__()


def reshape_batch(batch: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
  """Reshapes a batch to have the a leading dimension for the local devices."""
  leading_dims = [jax.local_device_count(), -1]
  return jax.tree.map(
      lambda x: np.reshape(x, leading_dims + list(x.shape[1:])), batch
  )


def evaluate(
    model: nn.Module,
    train_state: TrainState,
    eval_loader: grain.TfDataLoader,
    num_eval_steps: int = -1,
) -> EvalMetrics:
  """Evaluate the model on the given dataset."""
  logging.info("Starting evaluation.")
  eval_metrics = None
  with StepTraceContextHelper("eval", 0) as trace_context:
    # Use `iter` to reset the eval_loader before each evaluation.
    for step, batch in enumerate(iter(eval_loader)):
      batch = reshape_batch(batch)
      metrics_update = flax_utils.unreplicate(
          eval_step(model, train_state, batch)
      )
      eval_metrics = (
          metrics_update
          if eval_metrics is None
          else eval_metrics.merge(metrics_update)
      )
      if num_eval_steps > 0 and step + 1 == num_eval_steps:
        break
      trace_context.next_step()
  if eval_metrics is None:
    raise ValueError(f"Eval dataset {eval_loader} was empty.")
  return eval_metrics


def get_rng(seed: None | int | tuple[int, int]) -> np.ndarray:
  """Returns a JAX RNGKey."""
  if seed is None:
    # Case 1: No random seed given, use XManager ID.
    # All processes (and restarts) get exactly the same seed but every work unit
    # and experiment is different.
    work_unit = platform.work_unit()
    rng = (work_unit.experiment_id, work_unit.id)
  elif isinstance(seed, int):
    # Case 2: Single integer given.
    rng = (0, seed)
  else:
    # Case 3: tuple[int, int] given.
    if not isinstance(seed, (tuple, list)) or len(seed) != 2:
      raise ValueError(
          "Random seed must be an integer or tuple of 2 integers "
          f"but got {seed!r}"
      )
    rng = seed
  # JAX RNGKeys are arrays of np.uint32 and shape [2].
  return np.asarray(rng, dtype=np.uint32)


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> Mapping[str, Any]:
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.

  Returns:
    A dictionary that maps "train_state" to the TrainState and "train_iter" to
      the train iterator.
  """
  workdir = epath.Path(workdir)
  workdir.mkdir(parents=True, exist_ok=True)

  rng = get_rng(config.seed)
  logging.info("Using random seed %s.", rng)

  # Learning rate schedule.
  num_train_steps = input_pipeline.get_num_train_steps(config)
  steps_per_epoch = num_train_steps // config.num_epochs
  logging.info(
      "num_train_steps=%d, steps_per_epoch=%d", num_train_steps, steps_per_epoch
  )
  # We treat the learning rate in the config as the learning rate for batch size
  # 256 but scale it according to our batch size.
  global_batch_size = config.per_device_batch_size * jax.device_count()
  base_learning_rate = config.learning_rate * global_batch_size / 256.0
  learning_rate_fn = functools.partial(
      get_learning_rate,
      base_learning_rate=base_learning_rate,
      steps_per_epoch=steps_per_epoch,
      num_epochs=config.num_epochs,
      warmup_epochs=config.warmup_epochs,
  )

  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, optimizer, train_state = create_train_state(
      config,
      model_rng,
      input_shape=(8, 224, 224, 3),
      num_classes=1000,
      learning_rate_fn=learning_rate_fn,
  )

  # Set up checkpointing of the model and the input pipeline.
  checkpoint_manager = _get_checkpoint_manager(workdir)

  # Build input pipeline.
  rng, data_seed = jax.random.split(rng)
  data_seed = int(
      jax.random.randint(data_seed, [], minval=0, maxval=np.iinfo(np.int32).max)
  )
  train_loader, eval_loader = input_pipeline.create_datasets(config, data_seed)
  train_iter = iter(train_loader)

  # Retrieve data from previous checkpoints if possible.
  checkpointed_state = dict(train_state=train_state, train_iter=train_iter)
  if checkpoint_manager.latest_step() is not None:
    checkpointed_state = checkpoint_manager.restore(
        checkpoint_manager.latest_step(), items=checkpointed_state
    )
  train_state = checkpointed_state["train_state"]
  train_iter = checkpointed_state["train_iter"]

  # Distribute training.
  train_state = flax_utils.replicate(train_state)
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          optimizer=optimizer,
          learning_rate_fn=learning_rate_fn,
          weight_decay=config.weight_decay,
      ),
      axis_name="batch",
  )

  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.process_index() > 0
  )
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer
  )
  if jax.process_index() == 0:
    hooks += [
        report_progress,
        periodic_actions.Profile(num_profile_steps=5, logdir=workdir),
    ]
  train_metrics = None
  # Unreplicating from TPU is costly, so we only do it once at the start.
  initial_step = int(flax.jax_utils.unreplicate(train_state.step))
  with metric_writers.ensure_flushes(writer):
    # Steps are in interval [1, num_train_steps], not [0, num_train_steps - 1].
    for step in range(initial_step + 1, num_train_steps + 1):
      is_last_step = step == num_train_steps
      if step == 1:
        writer.write_hparams(dict(config))

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = reshape_batch(next(train_iter))
        train_state, metrics_update = p_train_step(
            train_state=train_state, batch=batch
        )
        metric_update = flax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None
            else train_metrics.merge(metric_update)
        )

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      for h in hooks:
        h(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        writer.write_scalars(step, train_metrics.compute())
        train_metrics = None

      if step % config.eval_every_steps == 0 or is_last_step:
        with report_progress.timed("eval"):
          train_state = merge_batch_stats(train_state)
          eval_metrics = evaluate(
              model, train_state, eval_loader, config.num_eval_steps
          )
        eval_metrics_cpu = jax.tree.map(np.array, eval_metrics.compute())
        writer.write_scalars(step, eval_metrics_cpu)

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          train_state = merge_batch_stats(train_state)
          checkpoint_manager.save(
              step,
              items=dict(
                  train_state=jax.tree.map(
                      np.array, flax_utils.unreplicate(train_state)
                  ),
                  train_iter=train_iter,
              ),
          )

  logging.info("Finishing training at step %d", num_train_steps)
  return checkpointed_state
