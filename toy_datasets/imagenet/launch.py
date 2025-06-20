r"""XManager script that launches ImageNet training on TPU.

Based on go/big_vision launcher -- see go/bv_launcher for list of cool features.

For example usage, please refer to the commands in the module docstrings of
`configs/default.py` and `configs/advanced.py` (requires go/brain-bashrc).
"""

from collections import abc
import importlib
import os
import types
from typing import Any

from absl import app
from absl import flags
import jax
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_abc  # Open-source would use xm_local
from xmanager.contrib.internal import artifacts as xmatf
from xmanager.contrib.internal import requirements_flag as xmreq
from xmanager.contrib.internal import tensorboard
from xmanager.contrib.internal import xm_jax

from google3.learning.deepmind.analysis.flatboard import flatboard as fb
from google3.learning.deepmind.python.adhoc_import import binary_import


PROJECT_PATH = "experimental/largesensormodels/toy_datasets/imagenet"

FLAGS = flags.FLAGS

# Experiment settings
_CONFIG = flags.DEFINE_string(
    "config", None,
    "A path to the configuration file to be ran, relative to 'configs/'.")
_SET_CONFIG = flags.DEFINE_multi_string(
    "set_config", [],
    "Config overrides, e.g. `--set_config num_epochs=1`. Can be specified "
    "multiple times. These overrides have precedence over anything that is set "
    "in the config file (including the values set in `sweep()`).")
_TAGS = flags.DEFINE_list(
    "tags", None, "Comma-separated tags, e.g. tag1,tag2")
_NOTE = flags.DEFINE_string(
    "note", "", "Optional free-form note for the experiment.")
_NAME = flags.DEFINE_string(
    "name", None, "Name of experiment. If unspecified, use config filename.")
_SWEEP = flags.DEFINE_boolean(
    "sweep", True,
    "Run the sweep if config file contains a function called `sweep()`.")
_WORKDIR = flags.DEFINE_string(
    "workdir", "/cns/{cell}-d/home/{author}/xm/{xid}",
    "Work directory where to store checkpoints and possibly other stuff. "
    "A few placeholders are available, see default value.")

_DUMP_HLO = flags.DEFINE_bool(
    "dump_hlo", False, "Dump all compilation HLO into workdir for debugging.")
_FLAX_PROFILE = flags.DEFINE_bool(
    "flax_profile", True,
    "Enable annotating every Flax module with a useful string in XProf. "
    "There are no runtime performance costs but JAX tracing will be ~10% "
    "slower, and Flax is more strict about function purity.")

# Core ccheduling/run settings
_PLATFORM = xmreq.DEFINE_requirements(
    "platform", None, "Accelerator specification. Eg.: cpu, v100=1, df=4x4)")
_PRIORITY = flags.DEFINE_integer(
    "priority", 200, "Use 200 for PROD, 115 for BATCH and 25 for FREEBIE.")
_IMPORTANCE = flags.DEFINE_enum(
    "importance", "normal", ["low", "normal", "high"], "Experiment importance.")

# Cell selection
_CELL = flags.DEFINE_string(
    "cell", None, "Which cell to run on. None means auto-select per work-unit. "
    "single or 1 means auto-select using a single best cell for all work units."
    " More details on the auto-selection process at go/xm-resource-selector.")
_CELLS_ONLY = flags.DEFINE_list(
    "cells_only", None, "Only select one of these cells.")
_CELLS_NOT = flags.DEFINE_list(
    "cells_not", None, "Do not use these cells under any circumstance.")

# Other misc scheduling/run settings:
_ADMINS = flags.DEFINE_list(
    "admins", None, "LDAPs of people who can modify the experiment.")
_MAX_PARALLEL_WORK_UNITS = flags.DEFINE_integer(
    "max_parallel_work_units", 1000,
    "Maximum number of work-units started in parallel for the hparam sweep.")
_MAX_FAILURES = flags.DEFINE_integer(
    "max_failures", 0,
    "Max. number of failures before the job will be killed. -1 == Unlimited. "
    "Setting this to a nonzero value can be very useful for batch/freebie jobs "
    "where there's frequent hardware failures, and we'd simply like to retry.")
_LOCAL_RAM_FS_GB = flags.DEFINE_integer(
    "local_ram_fs_gb", None, "How much RAM to use for local FS (in GiB).")


def main(argv: list[str]) -> None:
  del argv
  # There's many variants of the config file string we need.
  config = f"{PROJECT_PATH}/configs/{_CONFIG.value}"
  (config_name, arg, config_for_fileset, config_for_import,
   config_for_context) = get_config_paths(config)

  with xm_abc.create_experiment(
      experiment_title=_NAME.value or f"imagenet/{config_name}",
      settings=xm_abc.ExecutionSettings(
          max_parallel_work_units=_MAX_PARALLEL_WORK_UNITS.value,
          admin_users=_ADMINS.value,
      ),
  ) as xp:

    xp.context.annotations.add_tags(*parse_tags())
    xp.context.annotations.set_notes(_NOTE.value)
    xp.context.add_config_file(config_for_context)

    # We can set importance right away, before creating any WU.
    if _IMPORTANCE.value != "normal":
      xp.set_importance(xm.Importance[_IMPORTANCE.value.upper()])

    config_fileset = xm_abc.Fileset(files={config_for_fileset: "config.py"})

    device_type = list(_PLATFORM.value)[0]
    bazel_args = xm_abc.bazel_args.for_resource(device_type)
    # This is necessary for linking to succeed when using GPUs.
    # In some informal testing, it did not seem to slow down compilation of
    # non-GPU workloads, so we're always enabling it for simplicity.
    bazel_args = bazel_args + ("--define=cuda_compress=1",)

    # Common args for all jobs.
    exe_args = xm_jax.JaxFlags().flags()
    exe_args["jax_log_compiles"] = True

    # Note that `megacore_dense` is the same as `megacore`, but additioanlly
    # disables barnacore infeed, which eats ~1GB of extra memory.
    if device_type == "pf":
      exe_args["deepsea_chip_config_name"] = "megacore_dense"

    # If more binaries are added at any point, we should list them in this same
    # call, as that supposedly leads to much faster build times.
    [executable] = xp.package([
        xm.bazel_binary(
            label=f"//{PROJECT_PATH}:main",
            dependencies=[config_fileset],
            executor_spec=xm_abc.Borg.Spec(),
            # Need to wait for next build
            bazel_args=bazel_args,
            args=exe_args,
        ),
    ])

    executor = xm_abc.Borg(
        requirements=get_job_requirements(),
        logs_read_access_roles=["all"],
        # Autopilot has regularly caused us "pending forever" issues.
        autopilot_params=xm_abc.AutopilotParams(enabled=False),
        # This can help quite a bit for batch/freebie jobs:
        scheduling=xm_abc.BorgScheduling(
            max_task_failures=-1,
            max_per_task_failures=_MAX_FAILURES.value,
            task_failure_credit_period=3600,
        ),
    )

    # NOTE: Intentionally not removing /=% etc from name!
    workdir = _WORKDIR.value.format(  # pylint: disable=g-long-ternary
        cell=executor.requirements.location,
        name=xp.context.annotations.title,
        author=xp.context.creator,
        xid=xp.experiment_id,
    )

    # Checkpoint path includes work unit id. Construct it inside job generator.
    async def make_job(wu: xm_abc.XManagerWorkUnit,
                       *,
                       arg: str = arg,
                       tags: tuple[str, ...] = (),
                       **config_overrides: abc.Mapping[str, Any]) -> None:

      wu_workdir = os.path.join(workdir, str(wu.work_unit_id))

      # Build up the environment variables to be passed:
      env_vars = {}
      if _DUMP_HLO.value and wu_workdir:
        # See go/xla-debug-flags
        env_vars["XLA_FLAGS"] = f"--xla_dump_to={wu_workdir}"
      if _FLAX_PROFILE.value:
        # See http://google3/third_party/py/flax/configurations.py
        env_vars["FLAX_PROFILE"] = "true"  # Could also be experiment-wide.

      # The following fixes an issue with loading tfds splits containing "%":
      def sanitize(val: Any) -> Any:
        def sanitize_leaf(leaf: Any) -> Any:
          if isinstance(leaf, str):
            return leaf.replace("%", "%%")
          return leaf
        return jax.tree.map(sanitize_leaf, val)

      # Build up the commandline arguments to be passed:
      args = {
          "config": config_fileset.get_path(
              f"config.py:{arg}" if arg else "config.py",
              xm_abc.Borg.Spec()),
          **{f"config.{k}": sanitize(v) for k, v in config_overrides.items()},
      }
      args["workdir"] = wu_workdir
      for override in _SET_CONFIG.value:
        k, v = override.split("=", 1)
        args[f"config.{k}"] = v

      # A convenience check for a common mistake when porting old-style sweeps:
      if any(k.startswith("config.") for k in config_overrides):
        raise ValueError("Don't prefix sweeped vars with `config.` anymore.")

      wu.add(xm.Job(
          executable, args=args, env_vars=env_vars, executor=executor))
      if tags:
        wu.context.annotations.add_tags(*tags)
      if arg:
        wu.context.annotations.add_tags(f"arg:{arg}")

      xmatf.create_artifact(
          xp.experiment_id,
          xmatf.Type.ARTIFACT_TYPE_STORAGE2_BIGTABLE,
          f"/datatable/xid/{xp.experiment_id}/data:{wu.work_unit_id}",
          "datatable",
          wu.work_unit_id,
      )

    with binary_import.AutoGoogle3(verbose=False):
      config_module = importlib.import_module(config_for_import)

    read_and_do_sweep(config_module, arg,
                      lambda **kw: xp.add(make_job, args=kw))

    xmatf.create_artifact(
        xp.experiment_id, xmatf.Type.ARTIFACT_TYPE_FLATBOARD_URL,
        make_flatboard(config_module, arg, xp.experiment_id),
        "Flatboard")

    xmatf.create_artifact(
        xp.experiment_id, xmatf.Type.ARTIFACT_TYPE_DIRECTORY,
        workdir, "Workdir on CNS")

    xmatf.create_artifact(
        xp.experiment_id, xmatf.Type.ARTIFACT_TYPE_URL,
        f"https://datatable.corp.google.com/xid/{xp.experiment_id}/arrays",
        "Arrays and images datatable")

    # This is for an old-style TensorBoard job running on borg
    tensorboard.add_tensorboard_borg(
        xp, workdir,
        termination_delay_secs=3 * 24 * 60 * 60,  # Keep running for 72 hours
        executor=xm_abc.Borg())
    # This is basically an MLDash/TensorBoard exporter, for tensorboard.corp
    tensorboard.add_tensorboard_corp(
        xp, workdir,
        executor=xm_abc.Borg())


def get_job_requirements() -> xm.JobRequirements:
  """Gets requirements for a single job and optionally updates cell flag."""
  # Let's first figure out the requirements for the executor: either
  # explicit selection of everything, or auto cell-selection!
  common_reqs = dict(priority=_PRIORITY.value, **_PLATFORM.value)
  if _LOCAL_RAM_FS_GB.value:
    common_reqs["tmp_ram_fs"] = _LOCAL_RAM_FS_GB.value * xm.GiB
  if _CELL.value and _CELL.value not in ("single", "1"):
    requirements = xm.JobRequirements(location=_CELL.value, **common_reqs)
  else:
    # NOTE: in principle, we could add a constraint like:
    #       rs.Gpu(['V100', 'P100']) to allow flexible GPU selection too.
    constraints = [rs.Borg()]  # Use borg, not gcp.
    constraints += [rs.Location(only=_CELLS_ONLY.value,
                                exclude=_CELLS_NOT.value)]
    [requirements] = rs.select(rs.Job(
        constraints=constraints,
        requirements=xm.JobRequirements(**common_reqs),
    ))
  if _CELL.value in ("single", "1"):
    # Overwrite cell flag for correct handling of workdir artifact and tags.
    _CELL.value = requirements.location
  return requirements


def get_config_paths(config: str) -> tuple[str, str, str, str, str]:
  """Returns custom config paths."""
  # There's many variants of the config file string we need.
  # Keep in mind it has an optional free-form param: xp/lb/foo.py:B/16
  if ":" in config:
    config_file_path, arg = config.split(":", 1)
  else:
    config_file_path, arg = config, None
  config_name = os.path.basename(config_file_path)
  config_for_fileset = f"//{config_file_path}"
  config_for_import = f"google3/{config_file_path}"[:-len(".py")]
  config_for_context = config_file_path
  return (config_name, arg, config_for_fileset, config_for_import,
          config_for_context)


def parse_tags() -> list[str]:
  tags = _TAGS.value or []
  tags.append("brain-templates")
  tags.append(f"cell:{_CELL.value}")
  tags.append(f"p:{_PRIORITY.value}")
  tags.append("acc:" + xmreq.tag(_PLATFORM.value))
  tags.append(FLAGS.xm_resource_alloc)
  return tags


def maybe(param: str) -> list[str]:
  """Util to pass `param` as an argument only if it's not None."""
  return [param] if param else []


def read_and_do_sweep(config_module: types.ModuleType, arg: str,
                      add_wu_fn: abc.Callable[..., None]) -> None:
  """Reads sweep out of config file, and executes it."""

  if hasattr(config_module, "sweep") and _SWEEP.value:
    config_module.sweep(add_wu_fn, *maybe(arg))

  else:
    add_wu_fn()  # No sweep => Add a single work-unit.


def make_flatboard(config_module: types.ModuleType, arg: str, xid: int) -> None:
  """Attaches a flatboard to the experiment."""
  # The flatboard can either be a default dummy one, created from a list of
  # metrics defined in the config file, or a fancy flatboard layout created from
  # a function in the config file.

  if hasattr(config_module, "flatboard"):
    return config_module.flatboard(fb, xid, *maybe(arg)).save_url()

  if hasattr(config_module, "metrics"):
    metrics = config_module.metrics(*maybe(arg))
  else:
    metrics = ["train_loss"]

  # Normalize to (x_key, y_key) pairs
  metrics = [("step", m) if isinstance(m, str) else m for m in metrics]

  return fb.Dashboard(
      title=f"Dashboard for xid/{xid}", plots=[fb.Plot(  # pylint: disable=g-complex-comprehension
          title="",  # UI defaults to `y_key` if empty.
          x_key=x,
          y_key=y,
          transform=fb.DataTransform.INDIVIDUAL,  # Want to see each WID.
          hide_preemptions=False,
          enable_subsampling=False,
          enable_bucketing=False,  # Enable this if we want subsampling.
          data_groups=[fb.DataGroup(
              name=f"{xid}",
              queries=[fb.DataQuery(f"/datatable/xid/{xid}/data")],
          )],
      ) for x, y in metrics]
  ).save_url()


if __name__ == "__main__":
  app.run(main)
