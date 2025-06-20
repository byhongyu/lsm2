r"""Showcases advanced launcher features.

This config is based on `./default.py` (read that config first!), but makes use
of some advanced functionality:

1. Demonstrates how to provide `arg` argument from command line to `sweep()` and
   `get_config()` and how to pass it from `sweep()` to `get_config()`.
2. Specifies metric names to be shown in Flatboard.

Example usage:

xmanager launch experimental/largesensormodels/toy_datasets/imagenet/launch.py -- \
    --xm_resource_alloc=group:brain/brain-shared-user-xm \
    --platform=df=4x4 --config=advanced.py:trials=3

Example run: xid/49532669 (final accuracies 76., 76.5, 76.5; runtime ~2h10m)


Example usage with UPTC (useful for quick iteration & debugging):

experimental/largesensormodels/toy_datasets/imagenet/run_local.sh \
    --config=advanced.py:runlocal
"""

from collections import abc
from typing import Optional, Union

import ml_collections

from google3.experimental.largesensormodels.toy_datasets.imagenet.configs import common
from google3.experimental.largesensormodels.toy_datasets.imagenet.configs import default


def get_config(arg: Optional[str] = None) -> ml_collections.ConfigDict:
  """Config for training a ResNet50 on imagenet.

  Args:
    arg: An optional string argument that can be specified on the command line.
      For example `--config=advanced.py:runlocal` will pass the `"runlocal"`
      string, which in turn is parsed into `arg.runlocal=True`.

  Returns:
    A `ml_collections.ConfigDict` instance with the complete configuration for
    training the network.
  """

  config = default.get_config()

  arg = common.parse_arg(arg, runlocal=False)
  # It's often a good idea to also store `arg` values in `config` to make better
  # use of tools such as go/xolab to analyze config args.
  config.runlocal = arg.runlocal

  if arg.runlocal:
    config.warmup_epochs = 0.5
    config.num_epochs = 1

  return config


def metrics(
    arg: Optional[str] = None
) -> Union[abc.Sequence[str], abc.Sequence[tuple[str, str]]]:
  """Returns metrics to be shown in Flatboard."""
  del arg
  # Equivalent to return (("step", "train_loss"), ("step", "eval_loss"), ...)
  return (
      "train_loss", "eval_loss", "accuracy", "accuracy", "learning_rate",
      "steps_per_sec", "uptime",
  )


def sweep(add: abc.Callable[..., None], arg: Optional[str] = None) -> None:
  """Starts multiple work units with varying config args."""
  # If a user specifies `--config:advanced.py:runlocal` then this function is
  # called with `arg="runlocal"`. The function `get_config()` is called with
  # the same `arg="runlocal"` from the command line by default, but it's also
  # possible for `sweep()` to override that `get_config(arg)`. This makes it
  # possible to sweep over `arg` values as well.
  arg = common.parse_arg(arg, runlocal=False, trials=1)
  # It's possible to set sweep params like `--config=advanced.py:trials=3`
  for trial in range(arg.trials):
    add(
        # Note that `get_config(arg)` is different from `sweep(arg)`, so we
        # need to specify it here as well.
        arg=f"runlocal={arg.runlocal}",
        # Standard `config.trial=trial` override.
        trial=trial,
        # For later analysis, it's a good idea to also specify all arguments
        # here from `arg` (even more so when sweeping over them).
        runlocal=arg.runlocal,
        tags=(f"wu_{trial}",),
    )
