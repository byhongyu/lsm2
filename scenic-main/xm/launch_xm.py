r"""XManager script to launch Scenic experiments.

Usage:
  $ gxm third_party/py/scenic/google/xm/launch_xm.py \
    --config=third_party/py/scenic/projects/<project>/configs/default.py -t
    jd=2x2 -exp_name my_important_experiment

  The -t flag specifies the training platform in short form
  (jd=2x2 => JellyDonut TPU 2x2).
  If the `platform` string is empty the platform will be CPU. Otherwise, it must
  be a comma-separated list of valid GPU specs or a valid TPU spec.
  A GPU spec can be the GPU type (p100, v100) or <type>_<num_gpus>.
  The '_<num_gpus>' is optional and defaults to 1. For multiple GPU types the
  maximum <num_gpus> is used.
  A TPU spec must have the format <tpu_version>_<tpu_topology>, where
  <tpu_version> can be jellyfish, jf, dragonfish or df.
  Valid values of <tpu_topology> are defined in
  //production/borg/platforms-accelerators/jellyfish/fish.borg.

  Examples:
  p100 - Request a single GPU of type P100.
  gpu=4 - Request 4 GPUs, type auto-detected.
  p100=4 - Request 4 GPUs of type P100.
  p100=4,v100=4 - Request 4 GPUs of type P100 or V100.
  jd=1x1 - Request a single Jellyfish chip.
  df=4x4 - Request a 4x4 slice of a Dragonfish pod.
  pf=2x4x4 - Request a 2x4x4 slice of a Pufferfish pod.

  See the helper string for supported options.
  There are more flags to change priority and cell as well as the exp_name
  and working directory.

New JAX related flags:
- By default this script will run your job with --jax_tpu_async.
  --jax_tpu_async should improve performance but can cause OOM errors
  for models that are close to the available memory. You can turn it off and
  contact the JAX team for help.
- The script also accepts --xm_megacore to activate the Megacore feature on
  Pufferfish. This is currently off by default, but we recommend trying it if
  you are running on Pufferfish.
- If you are profiling Flax code you can also pass --flax_profile to get named
  traces in Xprof.

Requirements for projects:
- You need to define a Python binary that accepts --workdir and --config
  flags (--config must be a ml_collections.config_flags.DEFINE_config_file
  flag.)
- You should also define a binary rule for go/fragmented-python. If your Python
  binary is named :main, create a fragmented_py_binary_mpms with name
  :main_fragmented_mpms.
- Your config file must be a valid ConfigDict file and define a method
  `get_config()` that returns a ml_collections.ConfigDict.

The script follows the best practices outlined in go/bt-xm.
"""

from collections.abc import Iterable
import datetime
import getpass
import os
from typing import Any, Dict, Tuple

from absl import app
from absl import flags
from ml_collections import config_flags
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib.internal import artifacts
from xmanager.contrib.internal import tensorboard
from xmanager.contrib.internal import xm_jax
from xmanager.contrib.internal import xm_tf_data_service
from xmanager.vizier import vizier_abc

from google3.learning.deepmind.experiments.xbinder import xbinder_api
from google3.learning.deepmind.python.adhoc_import import binary_import
from google3.pyglib import file_util
from google3.pyglib import gfile
from google3.pyglib.flags.contrib import duration_flag


with binary_import.AutoGoogle3():
  from scenic.google.xm import launch_utils  # pylint: disable=g-import-not-at-top
  # TODO(mentzer,dehghani): Remove once cl/547722596 did land.
  from xmanager.contrib.internal import requirements_flag  # pylint: disable=g-import-not-at-top

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    None,
    "A path to a ConfigDict (needs to be relative to google3/), with config "
    "overrides in the form --config.seed=43.",
    lock_config=True,
)
_EXP_NAME = flags.DEFINE_string(
    "exp_name",
    "scenic",
    "XM experiment description.",
)
_TAGS = flags.DEFINE_list(
    "tags",
    "",
    "Comma-separated list of tags to add to default tags.",
)
_NOTES = flags.DEFINE_string(
    "notes",
    None,
    "XM experiment notes.",
)
_WORKDIR = flags.DEFINE_string(
    "workdir",
    "/xfile/scenic/{user}/rs=6.3/{exp_name}/{config_name}/{xid}",
    "Experiment path. The 'cell', 'config_name', 'exp_name' and 'xid' will be "
    "substituted if provided in the string.",
)
_CELL = flags.DEFINE_string(
    "cell",
    None,
    "Where to launch the training jobs.",
    short_name="c",
)
_CELL_EXCLUDE = flags.DEFINE_list(
    "cell_exclude",
    None,
    "Comma-separated list of _training_ cells to exclude when using auto cell"
    " selector.",
)
_CELL_INCLUDE = flags.DEFINE_list(
    "cell_include",
    None,
    "Comma-separated list of _training_ cells to include when using auto cell"
    " selector.",
)
_CNS_CELLS_INCLUDE = flags.DEFINE_list(
    "cns_cells_include",
    None,
    "Comma-separated list of _training_ cells to include when using auto CNS "
    "cell selector.",
)
_EVAL_CELL = flags.DEFINE_string(
    "eval_cell",
    None,
    "Where to launch the evaluation jobs. If not given, we default to --cell. "
    "If --cell is also not given, we will auto-select _a new cell_ for eval, "
    "to support different accelerators for training and inference. To colocate "
    "eval with training, pass --eval_cell=training and we will use the "
    "(possibly auto-selected) training cell.",
)
_PLATFORM = requirements_flag.DEFINE_requirements(
    "platform",
    None,
    "Platform to use for the training job. Can also specified (and swept) via"
    " config.platform. Note that either --platform or config.platform must"
    " be specified.",
    short_name="t",
)
_EVAL_PLATFORM = requirements_flag.DEFINE_requirements(
    "eval_platform",
    None,
    "Platform to use for the eval job.",
)
_PRIORITY = flags.DEFINE_integer(
    "priority",
    200,
    "Priority for the training job.",
    short_name="p",
)
_NUM_SLICES = flags.DEFINE_integer(
    "num_slices",
    1,
    "Number of MegaScale slices.",
    short_name="ns",
)
_EVAL_PRIORITY = flags.DEFINE_integer(
    "eval_priority",
    None,
    "Priority for the evaluation job. Defaults to the training priority",
)
_BINARY_TYPE = flags.DEFINE_enum(
    "binary_type",
    "fragmented_python",
    ["build_target", "ml_python", "fragmented_python"],
    "How to build and run the main binary.",
)
_BINARY = flags.DEFINE_string(
    "binary",
    "//third_party/py/scenic:main",
    "Binary to be executed.",
)
_EVAL_BINARY = flags.DEFINE_string(
    "eval_binary",
    None,
    "Binary for eval to run in parallel.",
)
_RUN_AS_USER = flags.DEFINE_string(
    "run_as_user",
    None,
    "The borguser under which the jobs should run.",
)
_MAX_PARALLEL_WORK_UNITS = flags.DEFINE_integer(
    "max_parallel_work_units",
    0,
    "Maximum number of work units to run in parallel."
    "Zero means run all workers in parallel.",
)

_SCHEDULING_TIME_QUANTUM = duration_flag.DEFINE_duration(
    "scheduling_time_quantum",
    None,
    "Scheduling time quantum to use.",
)

_DATASET_WORKERS = flags.DEFINE_integer(
    "dataset_workers",
    0,
    "Number of worker replicas for dataset service.",
    short_name="w",
)
_DATA_SERVICE_PRIORITY = flags.DEFINE_integer(
    "data_service_priority",
    None,
    "Priority for dataset workers if specified. Otherwise, 'priority' is used."
    "Only applies if 'dataset_workers' > 0.",
)

_TMP_RAM_FS_SIZE = flags.DEFINE_integer(
    "tmp_ram_fs_size",
    0,
    "To set a custom ram-disk size in MB. If <= 0, it will use the defaults of "
    "go/bt-xm, which is currently 500 MB.",
)
_JAX_TPU_ASYNC = flags.DEFINE_bool(
    "jax_tpu_async",
    True,
    "Run in JAX TPU asynchronous mode.",
)
_XLA_ENABLE_ASYNC_PPERMUTE = flags.DEFINE_enum(
    "xla_enable_async_ppermute",
    "true",
    ["false", "auto", "true"],
    "Enable async ppermute.",
)
_VALIDATE_HYPERPARAMETERS = flags.DEFINE_bool(
    "validate_hyperparameters",
    True,
    "Validate hyperparameters.",
)
_FLAX_PROFILE = flags.DEFINE_bool(
    "flax_profile",
    True,
    "Whether to use labelled traces for Flax. This is only available for Linen "
    "and makes Xprof easier to read. There are no runtime performance costs "
    "but JAX tracing will be slightly (think ~10%) slower.",
)
_BORG_MAX_PER_TASK_FAILURES = flags.DEFINE_integer(
    "borg_max_per_task_failures",
    0,
    "Borg will restart jobs the specified number of times if they fail.",
)
_BORG_TASK_FAILURE_CREDIT_PERIOD = flags.DEFINE_integer(
    "borg_task_failure_credit_period",
    0,
    "Borg will forget about a task failure after the specified number of "
    "seconds. Used in conjunction with borg_max_per_task_failures this can be "
    "used to automatically restart flaky jobs running at batch priority",
)
EVAL_RESTART_ALWAYS = flags.DEFINE_bool(
    "eval_restart_always",
    True,
    "If true, always restarts evaluator jobs. If false, uses default borg "
    "restart setting for TPU workers.",
)
_ALLOW_EXPERIMENTAL_DEPS = flags.DEFINE_bool(
    "allow_experimental_deps",
    False,
    "If true, allows having experimental dependencies in your binary. This is"
    "disabled by defauly by Pytype.",
)

_WORKSTREAMS = flags.DEFINE_list(
    "workstreams",
    [],
    "Comma-separated list of XM workstreams.",
)
_RESTART_XID = flags.DEFINE_integer(
    "restart_xid",
    None,
    "Restarts all stopped/failed units with the given xid using an updated "
    "binary. Updating the binary also requires passing "
    "the '--xm_allow_mismatching_citcs.' flag to the launch script.",
)

_VIZIER = flags.DEFINE_bool(
    "vizier",
    False,
    "If true, launches a Vizier study. Config file must have "
    "`get_study_config()` method defined.",
)

_IMPORTANCE = flags.DEFINE_enum(
    "importance",
    None,
    ["low", "normal", "high"],
    "Set importance of the work unit to the specified value at the start of "
    "the job. Note that this affects the entire resource alloc.",
)

_ADMIN_USERS = flags.DEFINE_list(
    "admin_users",
    [],
    "Comma-separated list of admin users.",
)
_ADD_TENSORBOARD = flags.DEFINE_bool(
    "add_tensorboard",
    True,
    "If true, adds tensorboard jobs (corp and borg).",
)
_SWEEP_NAME = flags.DEFINE_string(
    "sweep_name",
    None,
    "Optional hyperparameter sweep name used in configs with"
    " hp_sweep_factory()",
)
_ATTRIBUTION_URLS = flags.DEFINE_list(
    "attribution_urls",
    [],
    "Comma-separated list of attribution urls. More details are at"
    "http://go/gdm-experiment-attribution-guide",
)


flags.mark_flags_as_required([_CONFIG.name])
flags.mark_flags_as_mutual_exclusive([_CELL_INCLUDE.name, _CELL_EXCLUDE.name])

flags.FLAGS.set_default("xm_monitor_on_launch", False)


TRAIN_NAME = "train"


def create_job(
    experiment: xm_abc.XManagerExperiment,
    requirements: xm.JobRequirements,
    config_resource: xm.BinaryDependency,
    args: Dict[str, Any],
    env_vars: Dict[str, str],
) -> xm.Job:
  """Creates a train job."""
  executor = xm_abc.Borg(
      requirements=requirements,
      borg_user=_RUN_AS_USER.value,
      logs_read_access_roles=["all"],
      scheduling=xm_abc.BorgScheduling(
          max_per_task_failures=_BORG_MAX_PER_TASK_FAILURES.value,
          task_failure_credit_period=_BORG_TASK_FAILURE_CREDIT_PERIOD.value,
      ),
      # 5 minutes to prepare for preemption.
      stop_time=datetime.timedelta(minutes=5),  # 300s
  )

  bazel_args = xm_abc.bazel_args.cpu()
  if executor.requirements.accelerator in xm.TpuType:
    bazel_args = xm_abc.bazel_args.tpu()
  if executor.requirements.accelerator in xm.GpuType:
    bazel_args = xm_abc.bazel_args.gpu()
  if _ALLOW_EXPERIMENTAL_DEPS.value:
    bazel_args = bazel_args + ("--experimental_deps_ok",)

  if _BINARY_TYPE.value == "ml_python":
    adhoc_import_modules = ["google3.third_party.py.scenic"]
    [executable] = experiment.package([
        xm_abc.interpreter(
            script_path=_BINARY.value.replace(":", "/") + ".py",
            interpreter_mpm=xm_abc.ml_python(),
            adhoc_import_modules=adhoc_import_modules,
            dependencies=[config_resource],
            args=args,
            env_vars=env_vars,
        )
    ])

  else:
    label = _BINARY.value
    if _BINARY_TYPE.value == "fragmented_python":
      label += "_fragmented_mpms"
    [executable] = experiment.package([
        xm.bazel_binary(
            executor_spec=executor.Spec(),
            label=label,
            bazel_args=bazel_args,
            dependencies=[config_resource],
            args=args,
            env_vars=env_vars,
        )
    ])
  return xm.Job(executable=executable, executor=executor)


def create_eval_job(
    experiment: xm_abc.XManagerExperiment,
    requirements: xm.JobRequirements,
    config_resource: xm.BinaryDependency,
    args: Dict[str, Any],
    env_vars: Dict[str, str],
) -> xm.Job:
  """Creates an eval job that will run in parallel to training."""
  executor = xm_abc.Borg(
      requirements=requirements,
      borg_user=_RUN_AS_USER.value,
      logs_read_access_roles=["all"],
      scheduling=xm_abc.BorgScheduling(
          max_task_failures=-1, max_per_task_failures=-1
      ),
  )

  eval_args = {
      "eval_while_training": True,
      **args,
  }

  bazel_args = xm_abc.bazel_args.cpu()
  if executor.requirements.accelerator in xm.TpuType:
    bazel_args = xm_abc.bazel_args.tpu()
  if executor.requirements.accelerator in xm.GpuType:
    bazel_args = xm_abc.bazel_args.gpu()
  if _ALLOW_EXPERIMENTAL_DEPS.value:
    bazel_args = bazel_args + ("--experimental_deps_ok",)

  [executable] = experiment.package([
      xm.bazel_binary(
          executor_spec=executor.Spec(),
          label=_EVAL_BINARY.value,
          bazel_args=bazel_args,
          dependencies=[config_resource],
          args=eval_args,
          env_vars=env_vars,
      )
  ])
  return xm.Job(executable, executor)


def create_tf_data_service_jobs(
    experiment,
    cell: str,
    parent: str,
) -> Tuple[xm.JobGroup, xm_abc.RESTRICTED_BorgToken]:
  """Creates jobs for running the tf.data service."""
  tf_data_dispatcher, tf_data_worker = experiment.package([
      xm_tf_data_service.default_dispatcher_binary(),
      xm_tf_data_service.default_worker_binary(),
  ])
  data_service = xm_tf_data_service.TfDataService(
      experiment=experiment,
      worker_requirements=xm.JobRequirements(
          location=cell,
          priority=(
              _DATA_SERVICE_PRIORITY.value
              if _DATA_SERVICE_PRIORITY.value is not None
              else _PRIORITY.value
          ),
          replicas=_DATASET_WORKERS.value,
      ),
      fixed_replicas=True,
      base_dir=(
          f"/cns/{xm_tf_data_service.base_dir_cell()}-d/home/"
          f"{getpass.getuser()}/ttl=7d"
      ),
      parent_job=parent,
      dispatcher_executable=tf_data_dispatcher,
      worker_executable=tf_data_worker,
  )
  group, address = data_service.create_job_group()
  for job in group.jobs.values():
    if isinstance(job, xm.Job):
      executor = job.executor
      assert isinstance(executor, xm_abc.Borg)
      executor.borg_user = _RUN_AS_USER.value

  return group, address


def _platform_types(platforms) -> set[str]:
  """Returns the set of platform types.

  E.g. for [{}, {"pf": "2x2x2"}], this returns set("cpu", "pf").

  Args:
    platforms: Iterable of platforms dicts, see above.

  Returns:
    Set of all platforms.
  """
  result = set()
  for platform_dict in platforms:
    if not platform_dict:
      result.add("cpu")
    elif len(platform_dict) > 1:
      raise ValueError(f"Must have only one entry: {platform_dict}")
    else:
      result.add(next(iter(platform_dict.keys())))
  return result


def _select_cns_cell(target_cell: str, candidates: Iterable[str]):
  """Selects a cell from `candidates` close to `target_cell`.

  First, we try to find a cell on the same fabric, then in the same metro.

  Args:
    target_cell: What we want to be close to.
    candidates: Available cells to pick from.

  Returns:
    A cell that is close.

  Raises:
    ValueError: No cells in `candidates` are suitable!
  """
  if target_cell in candidates:
    return target_cell

  for locality in (rs.SameFabric(), rs.SameMetro()):
    try:
      _, requirements = rs.select(
          rs.Job(constraints=[rs.Location(only=[target_cell]), locality]),
          rs.Job(constraints=[rs.Location(only=candidates), locality]),
      )
    except rs.SelectionError:
      print(f"{locality}: No cells...")
    else:
      cell = requirements.location
      assert cell
      print(f"{locality}: Cell near {target_cell}={cell}")
      return cell

  raise ValueError(f"No cells in {candidates} are near `{target_cell}`")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if flags.FLAGS.xm_resource_alloc.startswith("group:brain/grand-vision"):
    if not _PLATFORM.value:
      # TODO(mentzer,dehghani): This is incompatible with config.platform
      # sweeps.
      raise ValueError("Need platform!")
    flags.FLAGS.xm_resource_alloc = launch_utils.get_preferred_resource_alloc(
        flags.FLAGS.xm_resource_alloc, _PLATFORM.value
    )

  config = config_flags.get_config_filename(flags.FLAGS["config"])
  (config_resource, config_ref, config_file_path, config_dir, config_name) = (
      launch_utils.get_config(config, _BINARY_TYPE.value)
  )
  # The config_string should probably come out of get_config directly.
  # Consider doing so when migrating to the new XManager API.
  if len(config_ref.rsplit(":")) == 2:
    config_string = config_ref.rsplit(":")[1]
  else:
    config_string = None

  exp_name = "_".join(_EXP_NAME.value.split())
  experiment_name = f"{exp_name}"
  if _RESTART_XID.value:
    experiment = xm_abc.get_experiment(_RESTART_XID.value)
  else:
    settings = xm_abc.ExecutionSettings(
        admin_users=_ADMIN_USERS.value,
        max_parallel_work_units=_MAX_PARALLEL_WORK_UNITS.value,
        scheduling_time_quantum=_SCHEDULING_TIME_QUANTUM.value,
    )
    experiment = xm_abc.create_experiment(
        experiment_name,
        attribution_urls=_ATTRIBUTION_URLS.value or None,
        settings=settings,
    )

  platforms = []
  platform_sweep = {None}
  if not _VIZIER.value:
    # Parse parameters to get platform strings.
    parameters = launch_utils.get_parameter_sweep(
        config_file_path,
        validate_hyperparameters=_VALIDATE_HYPERPARAMETERS.value,
        config_string=config_string,
        sweep_name=_SWEEP_NAME.value,
    )
    # Note that the config strings are hashable so we can build a set of unique
    # platforms.
    platform_sweep = {hparams.get("config.platform") for hparams in parameters}
    if platform_sweep and platform_sweep != {None}:
      print("Setting platforms given sweep...")
      platforms = [
          requirements_flag.parse_requirements_spec(platform)
          for platform in platform_sweep
      ]
      # TODO(mentzer,dehghani): We only support sweeps of the same type (eg
      # sweep topologies of pf).
      platform_types = _platform_types(platforms)
      if len(platform_types) > 1:
        raise ValueError(
            "For config.platform sweeps, all platforms must have the same"
            " type, since we assume all workunits can be in the same cell!"
            f" Got {platform_types}!"
        )
  if not platforms:  # Either vizier or no platform sweep.
    if not _PLATFORM.value:
      raise ValueError("Need --platform!")
    platforms = [_PLATFORM.value]

  assert len(platforms) >= 1  # Programmer error.

  with experiment:
    args = {
        "config": config_resource.get_path(config_ref, xm_abc.Borg.Spec()),
        "xprof_port": "%port_xprof%",
        "xla_enable_async_all_gather": "auto",
        "xla_enable_async_collective_permute": _XLA_ENABLE_ASYNC_PPERMUTE.value,
        "xla_tpu_rwb_fusion": "true",
        # We need the new scheduler features to get async ppermute to be
        # scheduled properly
        "xla_tpu_enable_all_experimental_scheduler_features": (
            _XLA_ENABLE_ASYNC_PPERMUTE.value
        ),
    }
    for k, v in config_flags.get_override_values(flags.FLAGS["config"]).items():
      args[f"config.{k}"] = v

    use_megacore = [*platforms[0].keys()][0] == "pf" and flags.FLAGS.xm_megacore

    exe_args = xm_jax.JaxFlags(
        jax_tpu_async=_JAX_TPU_ASYNC.value,
        jax_tpu_runtime=flags.FLAGS.jax_tpu_runtime,
        # `megacore_dense` is the same as `megacore`, but additionally
        # disables barnacore infeed, which eats ~1GB of extra memory.
        deepsea_chip_config_name="megacore_dense" if use_megacore else None,
    ).flags()
    exe_args["jax_log_compiles"] = True
    exe_args["g3pdb_port"] = "%port_g3pdb%"
    exe_args["brain_debug_port"] = "%port_debug%"

    num_slices = _NUM_SLICES.value
    if num_slices > 1:
      if exe_args.get("deepsea_chip_config_name") == "megacore":
        del exe_args["deepsea_chip_config_name"]

      address = "train_slice_00.get_job_bns_prefix()"
      exe_args["jax_controller_address"] = xm_abc.RESTRICTED_BorgToken(
          f'{address}+"/0:jax"'
      )
      exe_args["jax_num_tasks"] = xm_abc.RESTRICTED_BorgToken(
          f"replicas * {num_slices}"
      )
      exe_args["megascale_port"] = "%port_megascale%"
      exe_args["megascale_debug_port"] = "%port_megascaledebug%"
      exe_args["megascale_num_slices"] = num_slices
      exe_args["megascale_coordinator_address"] = xm_abc.RESTRICTED_BorgToken(
          f'{address}+"/0:megascale"'
      )
      exe_args["megascale_port_name"] = "megascale"
      exe_args["megascale_transport_type"] = "bamm"
      exe_args["megascale_abort_on_errors"] = True

    args.update(exe_args)

    env_vars = {"FLAX_PROFILE": str(_FLAX_PROFILE.value).lower()}

    def get_requirements(platform, cell: str | None):
      req_dict = dict(platform)
      req_dict["location"] = cell
      req_dict["priority"] = _PRIORITY.value
      if _TMP_RAM_FS_SIZE.value > 0:
        req_dict["tmp_ram_fs"] = _TMP_RAM_FS_SIZE.value * xm.MiB
      requirements = xm.JobRequirements(**req_dict)
      if requirements.location is None:
        assert cell is None  # Programmer error.
        constraints = [rs.Borg()]
        if exclude := _CELL_EXCLUDE.value:
          print("Not running in", exclude)
          constraints.append(rs.Location(exclude=exclude))
        elif include := _CELL_INCLUDE.value:
          print("Only running in", include)
          constraints.append(rs.Location(only=include))
        [requirements] = rs.select(
            rs.Job(requirements=requirements, constraints=constraints)
        )
      elif requirements.location == "viglobal":
        exclude = _CELL_EXCLUDE.value
        include = _CELL_INCLUDE.value
        if exclude:
          print("Not running in", exclude)
        if include:
          print("Only running in", include)
        [requirements] = rs.select(
            rs.Job(
                requirements=requirements,
                # constraints=[rs.Location(only=include, exclude=exclude)],
                constraints=[rs.Location(only=["metro:cbf"])],
            )
        )
      return requirements

    # We use the first platform to get the cell.
    main_requirements = get_requirements(platforms[0], _CELL.value)
    cell = main_requirements.location

    if allowed_cns_cells := _CNS_CELLS_INCLUDE.value:
      cns_cell = _select_cns_cell(cell, allowed_cns_cells)
    else:
      cns_cell = cell

    # Each experiment will use a unique folder for summary writing.
    workdir = _WORKDIR.value.format(
        cell=cns_cell,
        config_name=config_name,
        exp_name=exp_name,
        user=getpass.getuser(),
        xid=experiment.experiment_id,
    )

    tags = [
        config_dir,
        config_name,
        f"cell:{cell}",
        f"p:{_PRIORITY.value}",
        f"alloc:{flags.FLAGS.xm_resource_alloc}",
        "Scenic",
    ] + _TAGS.value
    for platform in platforms:
      tags.append(f"{requirements_flag.tag(platform)}")
    if cns_cell != cell:
      tags += [f"cns:{cns_cell}"]

    experiment.context.add_config_file(
        file_path=config, description="Experiment configurations"
    )
    annotations = experiment.context.annotations
    annotations.add_tags(*tags)
    if _NOTES.value:
      annotations.set_notes(_NOTES.value)

    jobs = {}
    jobs_to_always_run = []
    if _DATASET_WORKERS.value > 0:
      dataset_service_jobs, data_service_address = create_tf_data_service_jobs(
          experiment, cell, TRAIN_NAME
      )
      args["dataset_service_address"] = data_service_address
      jobs["dataset_service_jobs"] = dataset_service_jobs
      jobs_to_always_run.append("dataset_service_jobs")

    # We make one train job per platform, which is called eg "train_pf2x2x2".
    def get_train_name(platform) -> str:
      # Note that this must be a valid BCL name, go/bcl-basics#object-names.
      platform_key = requirements_flag.tag(platform).replace("=", "")
      return f"{TRAIN_NAME}-{platform_key}"

    for platform in platforms:
      if len(platforms) == 1 and platform_sweep == {None}:
        # To avoid ugly XM jobs, we stick to simple names if there is no
        # platform sweep.
        train_name_for_platform = TRAIN_NAME
        requirements = main_requirements
      else:
        train_name_for_platform = get_train_name(platform)
        requirements = get_requirements(platform, cell)

      jobs[train_name_for_platform] = create_job(
          experiment,
          requirements,
          config_resource,
          args,
          env_vars,
      )

    if _EVAL_BINARY.value is not None:
      req_dict = dict(_EVAL_PLATFORM.value)
      if custom_eval_cell := _EVAL_CELL.value:
        if custom_eval_cell == "training":
          # Re-use the (possibly auto-selected) training cell.
          req_dict["location"] = cell
        else:
          # User gave a cell, use it.
          req_dict["location"] = custom_eval_cell
      else:
        # Default to --cell. If no training cell is selected, we will
        # auto-select a new cell for eval.
        req_dict["location"] = _CELL.value
      req_dict["priority"] = (
          _EVAL_PRIORITY.value
          if _EVAL_PRIORITY.value is not None
          else _PRIORITY.value
      )
      if _TMP_RAM_FS_SIZE.value > 0:
        req_dict["tmp_ram_fs"] = _TMP_RAM_FS_SIZE.value * xm.MiB
      eval_requirements = xm.JobRequirements(**req_dict)
      if eval_requirements.location is None:
        [eval_requirements] = rs.select(
            rs.Job([rs.Borg()], requirements=eval_requirements)
        )
      elif eval_requirements.location == "viglobal":
        exclude = _CELL_EXCLUDE.value
        include = _CELL_INCLUDE.value
        if exclude:
          print("Eval not running in", exclude)
        if include:
          print("Eval only running in", include)
        [eval_requirements] = rs.select(
            rs.Job(
                requirements=eval_requirements,
                constraints=[rs.Location(only=include, exclude=exclude)],
            )
        )
      jobs["eval"] = create_eval_job(
          experiment=experiment,
          requirements=eval_requirements,
          config_resource=config_resource,
          args=args,
          env_vars=env_vars,
      )

    if _RESTART_XID.value:
      if len(platforms) > 1:
        # TODO(mentzer,dehghani): Have to find the right job.
        raise NotImplementedError("Restart not implemented for platform sweeps")
      for work_unit in experiment.work_units.values():
        work_unit.replace(xm.JobGroup(**jobs))
      return

    async def generate_jobs(work_unit: xm.WorkUnit, **hparams):
      hparams = hparams.copy()
      hparams["workdir"] = os.path.join(workdir, str(work_unit.work_unit_id))
      use_megascale_xla_runtime = _NUM_SLICES.value > 1

      if platform := hparams.get("config.platform"):
        train_name_for_job = get_train_name(
            requirements_flag.parse_requirements_spec(platform)
        )
      else:
        train_name_for_job = TRAIN_NAME

      if use_megascale_xla_runtime:
        assert len(jobs) == 1, "megascale support is limited to train only"
        job = jobs[train_name_for_job]
        megascale_jobs = {}
        executable = job.executable
        job_to_slice_id = {}
        group_args = {}
        for i in range(_NUM_SLICES.value):
          args = {
              "jax_task_id": xm_abc.RESTRICTED_BorgToken(
                  f'"\\"$(( %task% + {i} * " + replicas + " ))\\""'
              ),
              "megascale_slice_id": i,
          }
          args["megascale_debug_dir"] = f"{workdir}/megascale_debug"
          args.update(hparams)
          job_name = f"{train_name_for_job}_slice_{i:02d}"
          job_to_slice_id[job_name] = i
          group_args[job_name] = {"args": args}
          megascale_jobs[job_name] = xm.Job(
              executable=executable, executor=job.executor, args=job.args
          )
        work_unit.add(xm.JobGroup(**megascale_jobs), args=group_args)
      else:
        args = {}
        selected_jobs = {}
        for job_name in jobs_to_always_run:
          # These jobs do not get the hparams as arguments.
          selected_jobs[job_name] = jobs[job_name]
        for job_name in [train_name_for_job, "eval"]:
          if job_name in jobs:
            args[job_name] = {"args": hparams}
            selected_jobs[job_name] = jobs[job_name]
        assert selected_jobs  # Programmer error, no jobs selected!
        work_unit.add(xm.JobGroup(**selected_jobs), args=args)

    if _VIZIER.value:
      assert len(platforms) == 1  # Not supported
      study_config = launch_utils.get_vizier_study_config(config_file_path)
      study_factory = vizier_abc.NewStudy(study_config)
      experiment.add(
          vizier_abc.vizier_controller(
              generate_jobs,
              study_factory,
              num_parallel_work_units=_MAX_PARALLEL_WORK_UNITS.value,
          )
      )
    else:
      parameters = launch_utils.get_parameter_sweep(
          config_file_path,
          validate_hyperparameters=_VALIDATE_HYPERPARAMETERS.value,
          config_string=config_string,
          sweep_name=_SWEEP_NAME.value,
      )
      for hparams in parameters:
        experiment.add(generate_jobs, args=hparams)

    if _ADD_TENSORBOARD.value:
      tensorboard.add_tensorboard_borg(
          experiment,
          workdir,
          # Keep tensorboard_borg running for 72 hours.
          termination_delay_secs=3 * 24 * 60 * 60,
          executor=xm_abc.Borg(borg_user=_RUN_AS_USER.value),
      )
      tensorboard.add_tensorboard_corp(
          experiment,
          workdir,
          executor=xm_abc.Borg(borg_user=_RUN_AS_USER.value),
      )

  xid = experiment.experiment_id
  # Add an artifact with the workdir to the experiment.
  artifacts.create_artifact(
      xid=xid,
      artifact_type=artifacts.Type.ARTIFACT_TYPE_DIRECTORY,
      artifact=workdir,
      description="Workdir",
  )

  artifacts.create_artifact(
      xid=xid,
      artifact_type=artifacts.Type.ARTIFACT_TYPE_URL,
      artifact=f"https://datatable.corp.google.com/xid/{experiment.experiment_id}/images",
      description="Images datatable",
  )

  artifacts.create_artifact(
      xid=xid,
      artifact_type=artifacts.Type.ARTIFACT_TYPE_URL,
      artifact=f"https://datatable.corp.google.com/xid/{experiment.experiment_id}/arrays",
      description="Arrays datatable",
  )

  artifacts.create_artifact(
      xid=xid,
      artifact_type=artifacts.Type.ARTIFACT_TYPE_FLATBOARD_URL,
      artifact=launch_utils.make_flatboard(xid),
      description="Flatboard",
  )

  if _CELL.value is not None:
    google3_dir = file_util.FindGoogle3Dir(os.path.abspath("."))
    config_dir = os.path.join(workdir, "config_file")
    gfile.MakeDirs(config_dir, mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE)
    gfile.Copy(
        os.path.join(google3_dir, config_file_path),
        os.path.join(config_dir, config_name + f"_{xid}.py"),
        overwrite=True,
    )

    artifacts.create_artifact(
        xid=xid,
        artifact_type=artifacts.Type.ARTIFACT_TYPE_FILE,
        artifact=os.path.join(config_dir, config_name + f"_{xid}.py"),
        description="Experiment config",
    )

  for workstream in _WORKSTREAMS.value:
    ws = xbinder_api.Workstream(workstream)
    ws.add_experiment_id(experiment.experiment_id)

  # Set importance if specified. If possible, keep this call in the end of the
  # launcher, because it may fail if called too early (relative to exp.
  # launch) due to a race condition (see function's code for the pointers).
  launch_utils.set_importance(experiment, _IMPORTANCE.value)


def parse_flags_with_usage_google3(args):
  """Parse with adhoc import to support importing one config into another."""
  with binary_import.AutoGoogle3():
    return app.parse_flags_with_usage(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_flags_with_usage_google3)
