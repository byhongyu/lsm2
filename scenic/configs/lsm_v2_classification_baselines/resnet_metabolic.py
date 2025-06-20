r"""A config to train a ResNet on metabolic data.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
gxm experimental/largesensormodels/scenic/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/lsm_v2_classification_baselines/resnet_metabolic.py \
--platform=glp_1x1 \
--exp_name=lsm_supervised_metabolic_resnet \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_pool=plat-dev-dynamic \
--xm_resource_alloc=group:plat-dev-dynamic/h2o-ai-gqm-quota \
--xm_use_developer_builds_in_prod=false \
--cell=viglobal \
--priority=200

Note:
--priority=119 for Batch priority, --priority=25 for Freebie priority.
--TODO should work for this model too.
--viglobal is the cell for auto-selection.

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/lsm_v2_classification_baselines/resnet_metabolic.py:runlocal
"""


from typing import Optional  # pylint: disable=unused-import
import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants  # pylint: disable=unused-import


# To set constants.
# 1) Dataset / task variables.
DATASET_NAME = 'metabolic_tfrecords_24h_missingness_80'
TASK_NAME = [
    'hypertension_binary',
    'homa_ir_binray',
    'hyperlipidemia_binary',
    'diabetes_binary',
    'anxiety_binary',
    'cardiovascular_binary',
    'respiratory',
]

CACHE_DATASET = True
TRAIN_DATA_SIZE = None
USE_DATETIME_FEATURES = False
USE_TRAIN_AUGMENTATIONS = [False]
TRAIN_AUGMENTATIONS = ['noise']
SHUFFLE_SEED = 42
SHUFFLE_BUFFER_SIZE = 10_000

# 2) Model variables.
VARIANT = [
    'metadata_encoded_resnet',
    'resnet'
]
NUM_LAYERS = [18, 50]
# NUM_LAYERS = [200]  # Needs glp_2x2
NUM_FILTERS = 64
METADATA_ENCODING_FN = 'concat'

# 3) Loss variables.
LOSS_FN = ['balanced_softmax_loss']  # ['weighted_softmax_cross_entropy']
WEIGHTED_LOSS = [False]  # whether or not to use label weights
LABEL_SMOOTHING = [None]

# 3) Training / eval variables.
BATCH_SIZE = 1024
NUM_TRAIN_STEPS = 500

# Optimizer variables.
LRS = [5e-3]
WEIGHT_DECAYS = [1e-4]

# 4) Logging variables.
LOG_EVAL_SUMMARY_STEPS = 25  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = 25  # LOG_EVAL_SUMMARY_STEPS * 5
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)

  # 1. Experiment.
  config = ml_collections.ConfigDict()
  config.experiment_name = f'resnet-supervised-{DATASET_NAME}'
  config.shuffle_seed = SHUFFLE_SEED

  # 2. Dataset.
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = DATASET_NAME
  config.dataset_configs.cache_dataset = CACHE_DATASET
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = (
      None if runlocal else SHUFFLE_BUFFER_SIZE
  )

  config.dataset_configs.relative_time_window = None
  config.dataset_configs.task_name = TASK_NAME[0]

  # 3. Model.
  config.model_name = VARIANT[0]
  config.model = ml_collections.ConfigDict()
  config.model.patcher_config = ml_collections.ConfigDict()
  config.model.patcher_config.patchsize = None  # No patching is ResNet.
  config.model.num_layers = NUM_LAYERS[0]
  config.model.num_filters = NUM_FILTERS
  config.model.metadata_encoding_fn = METADATA_ENCODING_FN

  # 4. Model Loss
  config.classification_loss = ml_collections.ConfigDict()
  config.classification_loss.loss_name = LOSS_FN[0]
  config.classification_loss.label_smoothing = LABEL_SMOOTHING[0]
  config.classification_loss.weighted_loss = WEIGHTED_LOSS[0]

  # 5. Datetime features.
  config.use_datetime_features = USE_DATETIME_FEATURES

  # Training.
  config.trainer_name = 'lsm_supervised_trainer'
  config.batch_size = 8 if runlocal else BATCH_SIZE
  config.num_training_steps = 10 if runlocal else NUM_TRAIN_STEPS
  config.steps_per_eval = 10 if runlocal else None
  config.log_eval_steps = LOG_EVAL_SUMMARY_STEPS
  config.log_summary_steps = LOG_EVAL_SUMMARY_STEPS
  config.rng_seed = 42
  config.use_train_augmentations = USE_TRAIN_AUGMENTATIONS[0]
  config.train_augmentations = TRAIN_AUGMENTATIONS

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
      # Task
      hyper.sweep('config.dataset_configs.task_name', TASK_NAME),

      # Model
      hyper.sweep('config.model_name', VARIANT),
      hyper.sweep('config.model.num_layers', NUM_LAYERS),

      # Loss
      hyper.sweep(
          'config.classification_loss.label_smoothing', LABEL_SMOOTHING
      ),
      hyper.sweep('config.classification_loss.weighted_loss', WEIGHTED_LOSS),

      # Training / Optimizer
      hyper.sweep('config.schedule.all.lr_configs.base_learning_rate', LRS),
      hyper.sweep('config.optimizer.weight_decay', WEIGHT_DECAYS),
      hyper.sweep('config.use_train_augmentations', USE_TRAIN_AUGMENTATIONS),
  ])
