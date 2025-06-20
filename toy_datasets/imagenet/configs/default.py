r"""A config for training ResNet-50 on ImageNet.

Example usage with XManager:

xmanager launch experimental/largesensormodels/toy_datasets/imagenet/launch.py -- \
    --xm_resource_alloc=group:brain/brain-shared-user-xm \
    --platform=jd=2x2 --config=default.py

Example run: xid/49532144 (final accuracy 76.5, runtime ~11h)


Example usage with UPTC (useful for debugging):

experimental/largesensormodels/toy_datasets/imagenet/run_local.sh \
    --config=default.py
"""

from collections import abc

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Default config for training a ResNet50 on imagenet."""

  config = ml_collections.ConfigDict()

  # Possible values (from train.py): resnet18, resnet50.
  config.model_name = "resnet50"
  # This is the learning rate for batch size 256. The code scales it linearly
  # with the batch size. This is common for ImageNet and SGD.
  config.learning_rate = 0.1
  config.learning_rate_schedule = "cosine"
  config.warmup_epochs = 5.0
  config.sgd_momentum = 0.9
  config.weight_decay = 0.0001
  config.num_epochs = 90.0
  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs.
  config.num_train_steps = -1
  # Evaluates for a full epoch if num_eval_steps==-1. Set to a smaller value for
  # fast iteration when running train.train_and_eval() from a Colab.
  config.num_eval_steps = -1
  config.per_device_batch_size = 64
  # If batches should be added to evaluate the entire dataset.
  config.eval_pad_last_batch = True

  config.log_loss_every_steps = 500
  config.eval_every_steps = 5000
  config.checkpoint_every_steps = 5000

  # Single integer or tuple. If None will use (XManager ID, work unit).
  config.seed = None

  config.trial = 0  # Dummy for repeated runs.
  return config


# By default, the launcher calls `sweep()`.
# To disable the sweep, the `sweep()` function can be commented (or renamed),
# or the flag `--nosweep` can be specified to the launcher.
def sweep(add: abc.Callable[..., None]):
  """Starts multiple work units with varying config args."""
  for trial in range(1):
    # `trial=1` will set `config.trial=1`
    # use `add(**{'x.y.z': 1})` to set nested `config.x.y.z=1`
    add(trial=trial)
