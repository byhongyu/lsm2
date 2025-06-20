r"""A config to train a TinyShallow ViT MAE on LSM dataset.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
gxm third_party/py/scenic/google/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/supervised_lsm_simplecnn.py \
--platform=vlp_2x2 \
--exp_name=lsm_supervised_tier2_simplecnn \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_alloc=group:mobile-dynamic/h2o-ai-gqm-quota \
--xm_use_developer_builds_in_prod=false \
--priority=200

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/supervised_lsm_simplecnn.py:runlocal
"""


from typing import Optional  # pylint: disable=unused-import
import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants  # pylint: disable=unused-import


# To set constants.
# 1) Dataset variables.
DATASET_NAME = 'lsm_300min_600_activities_9class_subset'
CACHE_DATASET = True
TRAIN_DATA_SIZE = None
USE_DATETIME_FEATURES = False
USE_TRAIN_AUGMENTATIONS = True
TRAIN_AUGMENTATIONS = ['noise']
SHUFFLE_SEED = 42

LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = None  #1.5

# 2) Training / eval variables.
BATCH_SIZE = 128
NUM_TRAIN_STEPS = 5000
LRS = [1e-5, 1e-4, 1e-3]
WEIGHT_DECAYS = [1e-4, 1e-3]

# 3) Logging variables.
LOG_EVAL_SUMMARY_STEPS = 100  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = 100  # LOG_EVAL_SUMMARY_STEPS * 5
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)

# Model variant
VARIANT = 'simplecnn'
NUM_FILTERS = [20, 10, 10]
KERNEL_SIZES = [3, 3]


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)

  # Experiment.
  config = ml_collections.ConfigDict()
  config.experiment_name = f'electrodes-supervised-{DATASET_NAME}'
  config.dataset_name = f'lsm_prod/{DATASET_NAME}'
  config.shuffle_seed = SHUFFLE_SEED

  # Dataset.
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = f'lsm_prod/{DATASET_NAME}'
  config.dataset_configs.num_classes = None
  config.dataset_configs.train_split = 'train'  # train data split
  config.dataset_configs.train_num_samples = TRAIN_DATA_SIZE  # train sample
  # eval data split - note: this split is used for validation and test.
  config.dataset_configs.eval_split = 'test[:64]' if runlocal else 'test'
  config.dataset_configs.cache_dataset = CACHE_DATASET
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  config.model_name = 'simplecnn'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = tuple([1, 1])
  config.model.num_filters = NUM_FILTERS
  config.model.kernel_sizes = KERNEL_SIZES

  # Datetime features.
  config.use_datetime_features = USE_DATETIME_FEATURES

  # Training.
  config.trainer_name = 'lsm_supervised_trainer'
  config.batch_size = 8 if runlocal else BATCH_SIZE
  config.num_training_steps = NUM_TRAIN_STEPS
  config.log_eval_steps = LOG_EVAL_SUMMARY_STEPS
  config.log_summary_steps = LOG_EVAL_SUMMARY_STEPS
  config.rng_seed = 42
  config.use_train_augmentations = USE_TRAIN_AUGMENTATIONS
  config.train_augmentations = TRAIN_AUGMENTATIONS

  config.label_smoothing = LABEL_SMOOTHING
  config.max_grad_norm = MAX_GRAD_NORM

  sched = ml_collections.ConfigDict()
  sched.re = '(.*)'
  sched.lr_configs = ml_collections.ConfigDict()
  sched.lr_configs.learning_rate_schedule = 'compound'
  sched.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  sched.lr_configs.total_steps = NUM_TRAIN_STEPS
  sched.lr_configs.steps_per_cycle = sched.lr_configs.total_steps
  sched.lr_configs.warmup_steps = int(NUM_TRAIN_STEPS * 0.05)
  sched.lr_configs.base_learning_rate = LRS[0]
  config.schedule = ml_collections.ConfigDict({'all': sched})

  # *Single* optimizer.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scale_by_adam'
  # optim.optax = dict(mu_dtype='bfloat16')
  optim.optax_configs = ml_collections.ConfigDict({  # Optimizer settings.
      'b1': 0.9,
      'b2': 0.95,
  })
  config.optax = dict(mu_dtype='bfloat16')
  optim.max_grad_norm = 1.0
  optim.weight_decay = WEIGHT_DECAYS[0]
  optim.weight_decay_decouple = True
  config.optimizer = optim

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = LOG_CHECKPOINT_STEPS
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.max_checkpoints_to_keep = MAX_NUM_CHECKPOINTS
  # BEGIN GOOGLE-INTERNAL
  if runlocal:
    # Current implementation fails with UPTC.
    config.count_flops = False
  # END GOOGLE-INTERNAL

  return config


# BEGIN GOOGLE-INTERNAL
def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  return hyper.product([
      hyper.sweep('config.schedule.all.lr_configs.base_learning_rate', LRS),
      hyper.sweep('config.optimizer.weight_decay', WEIGHT_DECAYS),
  ])
