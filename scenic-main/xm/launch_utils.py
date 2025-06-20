r"""Utilities for xm launch scripts.

Based on the XManager launch script of Flax ImageNet example.
"""

from importlib import import_module  # pylint: disable=g-importing-member

import os
import time
from typing import Dict, Optional, Tuple, Union
import ml_collections
from xmanager import xm
from xmanager import xm_abc

from google3.learning.deepmind.analysis.flatboard import flatboard as fb
from google3.learning.deepmind.python.adhoc_import import binary_import
from google3.learning.deepmind.xmanager import hyper
from google3.learning.vizier.service.client import pyvizier


def get_config(
    config_flag,
    binary_type: Optional[str] = None
) -> Tuple[Union[xm_abc.Fileset, xm_abc.BazelDependency], str, str, str, str]:
  """Gets the resource for the config and the relative path to it the config."""
  # Create configuration resource.
  # go/config-flags#parameterising-the-get-config-function
  # `config_file_path`: Path to config to define the LocalFile resource
  # `config_ref`: Config w/ optional parametrisation; input to LocalFile.path_to
  config_file_path = config_flag.rsplit(':', 1)[0]
  if binary_type == 'build_target':
    # Find parent configs directory
    parent_path, dir_name = os.path.split(config_file_path)
    while dir_name != 'configs':
      if not dir_name or not parent_path or parent_path == dir_name:
        raise ValueError(
            'Expected config file to be under a `configs` directory, but no '
            'parent configs directory was found.')
      parent_path, dir_name = os.path.split(parent_path)
    config_base_dir = os.path.join(parent_path, dir_name)
    config_mpm = '//' + config_base_dir + ':config_files_mpm'
    config_resource = xm_abc.BazelDependency(config_mpm)
    config_ref = os.path.relpath(config_flag, start=config_base_dir)
  else:
    config_ref = os.path.basename(config_flag)
    config_resource = xm_abc.Fileset(
        {config_file_path: os.path.basename(config_file_path)})
  config_name = os.path.basename(config_flag).rstrip('.py')
  config_dir = os.path.basename(os.path.dirname(config_flag))
  return config_resource, config_ref, config_file_path, config_dir, config_name


def import_config_module(config_file_path):
  """Import config module."""
  # Loading here from file location avoids importing parent packages.
  # Importing //third_party/py/flax/__init__.py would fail because the XM CLI
  # does not include all Flax dependencies.
  if not config_file_path.startswith('google3'):
    config_file_path = os.path.join('google3', config_file_path)
  config_file_path = config_file_path.removesuffix('.py')
  with binary_import.AutoGoogle3():
    config_module = import_module(config_file_path)
  return config_module


def load_config(
    config_file_path,
    config_string: Optional[str] = None) -> ml_collections.ConfigDict:
  """Load configuration as ConfigDict."""
  config_module = import_config_module(config_file_path)
  if hasattr(config_module, 'get_config'):
    if config_string is None:
      return config_module.get_config()
    else:
      return config_module.get_config(config_string)
  else:
    raise AttributeError(
        f'config_module has no get_config() function: {config_file_path}')


def get_parameter_sweep(config_file_path,
                        validate_hyperparameters=True,
                        config_string: Optional[str] = None,
                        sweep_name: Optional[str] = None):
  """Gets the parameter sweep by calling `get_hyper` on the config module.

  The optional validation of hyperparameters is done immediately, before
  launching the experiment on borg.

  Args:
    config_file_path: path to config file.
    validate_hyperparameters: Boolean determining of hyperparameters should be
      validated or not.
    config_string: Optional config_string, which is an argument of get_config().
    sweep_name: Optional sweep_fn, which is an argument of hp_sweep_factory().

  Returns:
    hyperparameter sweep.
  """
  config_module = import_config_module(config_file_path)
  if hasattr(config_module, 'get_hyper'):
    hyper_params = config_module.get_hyper(hyper)
    if validate_hyperparameters:
      config = load_config(config_file_path, config_string)
      # Assume name of DEFINE_config_file flag is 'config'.
      hyper.assert_are_compatible(hyper_params, {'config': config})
    return hyper_params
  elif hasattr(config_module, 'hp_sweep_factory') and sweep_name:
    hyper_params = config_module.hp_sweep_factory(sweep_name)(hyper)
    if validate_hyperparameters:
      config = load_config(config_file_path, config_string)
      # Assume name of DEFINE_config_file flag is 'config'.
      hyper.assert_are_compatible(hyper_params, {'config': config})
    return hyper_params
  return hyper.product([])


def get_vizier_study_config(config_file_path):
  """Gets the Vizier study config calling `config_module.get_study_config`."""
  config_module = import_config_module(config_file_path)
  if hasattr(config_module, 'get_study_config'):
    return config_module.get_study_config(pyvizier)
  raise ValueError(f'get_study_config() is not defined in {config_file_path}.')


def get_preferred_resource_alloc(xm_resource_alloc: str,
                                 platform: Dict[str, str]) -> str:
  """Checks or create the preferred resource alloc for the executable.

  In Grand Vision we have multiple XM resource allocation that work globally
  (any cell) but are restricted to a single resource type. This is important
  in large groups due to the way XManager handles going above floor.

  Args:
    xm_resource_alloc: Resource allocation that is passed by user.
    platform: Platform for the experiment.

  Returns:
    A string of the form `group:brain/grand-vision-xm-df`.
  """
  if xm_resource_alloc != 'group:brain/grand-vision-xm':
    print('\nIf you are a member of the Grand Vision team it is recommended '
          "to set XM_DEFAULT_RESOURCE_ALLOC='group:brain/grand-vision-xm' "
          'in your ~/.bashrc and the script will choose the correct resource '
          'allocation for you.\n')

  acc = xm.JobRequirements(**platform).accelerator
  platform_to_resource_type = {
      xm.ResourceType.JELLYFISH: 'jf',
      xm.ResourceType.JELLYDONUT: 'jd',
      xm.ResourceType.DRAGONFISH: 'df',
      xm.ResourceType.DRAGONDONUT: 'dd',
      xm.ResourceType.PUFFERFISH: 'pf',
      xm.ResourceType.PUFFYLITE: 'pl',
      xm.ResourceType.VIPERLITE: 'vl',
      xm.ResourceType.VIPERLITE_POD: 'vlp',
      xm.ResourceType.VIPERFISH: 'vf',
  }

  if acc in xm.GpuType:
    resource_type = 'gpu'
  elif acc in platform_to_resource_type:
    resource_type = platform_to_resource_type[acc]
  else:
    return xm_resource_alloc

  gv_alloc = f'group:brain/grand-vision-xm-{resource_type}'
  if xm_resource_alloc == 'group:brain/grand-vision-xm-bets':
    return xm_resource_alloc
  if (xm_resource_alloc.startswith('group:brain/grand-vision-xm-')
      and xm_resource_alloc != gv_alloc):
    # If the user passed the type of resource via `xm_resource_alloc` Flag, make
    # sure it's aligned with the type of resource they asked for using
    # `platform` flag.
    raise ValueError(f'Wrong allocation, use "{xm_resource_alloc}" instead.')
  return gv_alloc


def set_importance(experiment, importance):
  """Set experiment's importance."""
  if importance:
    # See https://yaqs.corp.google.com/eng/q/6004157220257792.
    # tl;dr there is a race condition, so we try to set importance 10 times.
    num_attempts, success = 10, False
    for _ in range(num_attempts):
      try:
        experiment.set_importance(importance)
        success = True
      except:  # pylint: disable=bare-except
        time.sleep(1)
    if not success:
      print('Setting importance failed!')


def make_flatboard(xid, metrics=None):
  """Attaches a flatboard to the experiment."""
  # The flatboard can either be a default dummy one, created from a list of
  # metrics defined in the config file, or a fancy flatboard layout created from
  # a function in the config file.
  # This code is borrowed from bv launcher:
  #   google3/third_party/py/big_vision/launch.py

  metrics = metrics or ['train_loss']

  # Normalize to (x_key, y_key) pairs
  metrics = [('step', m) if isinstance(m, str) else m for m in metrics]

  return (
      fb.Dashboard(
          title=f'Dashboard for {xid}',
          plots=[
              fb.Plot(  # pylint: disable=g-complex-comprehension
                  title='',  # UI defaults to `y_key` if empty.
                  x_key=x,
                  y_key=y,
                  # Want to see each WID.
                  transform=fb.DataTransform.INDIVIDUAL,
                  hide_preemptions=False,
                  enable_subsampling=False,
                  enable_bucketing=False,  # Enable this if we want subsampling.
                  data_groups=[
                      fb.DataGroup(
                          name=f'{xid}',
                          queries=[fb.DataQuery(f'/datatable/xid/{xid}/data')],
                      )
                  ],
              )
              for x, y in metrics
          ],
      )
      .save_url()
      .split('/revisions/')[0]
  )
