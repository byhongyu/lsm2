r"""A config to pretraining train a Small ViT MAE on LSM V2 dataset.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
xmanager launch experimental/largesensormodels/scenic/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/lsm_v2_pretraining/mae_lsm_v2_v1_style_baseline.py \
--platform=glp_8x16 \
--exp_name=lsm_v2_v1_style_baseline \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_pool=plat-dev-dynamic \
--xm_resource_alloc=group:plat-dev-dynamic/h2o-ai-gqm-quota \
--xm_use_developer_builds_in_prod=false \
--cell=viglobal \
--priority=200

Note:
--priority=119 for Batch priority, --priority=25 for Freebie priority.
--glp_4x8 should work for this model too.
--viglobal is the cell for auto-selection.

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/lsm_v2_pretraining/mae_lsm_v2_v1_style_baseline.py:runlocal
"""

import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils.patcher_config import Patcher_Config
from google3.experimental.largesensormodels.scenic.trainers.masking.masker_config import MaskStrategy_Config, Masker_Config
from google3.experimental.largesensormodels.scenic.utils import config_constants
from google3.experimental.largesensormodels.scenic.utils import predefined_configs

# To set constants.
# 1) Dataset variables.
# TRAIN_DATASET_NAME = [
#     # 'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.2_timestamp_202504110407_doublethreshold_False',
#     # 'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504110538_doublethreshold_False'
#     'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.8_timestamp_202504110551_doublethreshold_False',
# ]

TRAIN_DATASET_NAME = [
    'lsm_v2_train_only_sessions_10000_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504172035_doublethreshold_False',
    'lsm_v2_train_only_sessions_1000_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504171841_doublethreshold_False',
    'lsm_v2_train_only_sessions_100_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504171758_doublethreshold_False',
]


VALID_DATASET_NAME = config_constants.VALID_DATASET_NAME
LSM_PREDEFINED_CONFIGS = predefined_configs.LSM_PREDEFINED_CONFIGS
LOSS_IGNORE_IMPUTATION = [True]
CACHE_DATASET = True
# TRAIN_DATA_SIZES = [1_000, 10_000, 100_000, 1_000_000, 1_601_088]
TRAIN_DATA_SIZES = [-1]
USE_DATETIME_FEATURES = False
USE_TRAIN_AUGMENTATIONS = [False]
TRAIN_AUGMENTATIONS = ['stretch', 'flip', 'noise']
SHUFFLE_SEED = 42
SHUFFLE_BUFFER_SIZE = 250_000

# 2) Training / eval variables.
BATCH_SIZE = 1024
NUM_TRAIN_STEPS = 100_000
LRS = [5e-3]
WEIGHT_DECAYS = [1e-4]

# 3) Logging variables.
LOG_EVAL_SUMMARY_STEPS = NUM_TRAIN_STEPS / 10  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = NUM_TRAIN_STEPS / 10  # LOG_EVAL_SUMMARY_STEPS * 5
LOG_TRAIN_SUMMARY_STEPS = NUM_TRAIN_STEPS / 100
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)
ENABLE_DUMP_MODE = False

# Model variant
VARIANT = 'S'

LOSS_ONLY_MASKED_TOKENS = True

# Downstream Tasks.

# Linear probe eval.
LINEAR_PROBE_USE_TRAIN_AUGMENTATIONS = False
LINEAR_PROBE_TRAIN_AUGMENTATIONS = ['noise']


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)

  # Experiment.
  config = ml_collections.ConfigDict()
  if runlocal:
    config.runlocal = True
  else:
    config.runlocal = False

  config.experiment_name = f'LSM V2-{TRAIN_DATASET_NAME[0]}'
  config.shuffle_seed = SHUFFLE_SEED
  config.loss_ignore_imputation = LOSS_IGNORE_IMPUTATION[0]

  # Dataset.
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = TRAIN_DATASET_NAME[0]
  config.dataset_configs.valid_dataset = VALID_DATASET_NAME
  config.dataset_configs.num_classes = None
  config.dataset_configs.train_split = 'train'  # train data split
  config.dataset_configs.train_num_samples = TRAIN_DATA_SIZES[0]  # train sample
  config.dataset_configs.eval_split = 'valid'
  config.dataset_configs.eval_num_samples = 64 if runlocal else None
  config.dataset_configs.cache_dataset = CACHE_DATASET
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = (
      256 if runlocal else SHUFFLE_BUFFER_SIZE
  )
  config.enable_dump_mode = ENABLE_DUMP_MODE
  # Model.
  version = VARIANT

  version = 'Deb' if runlocal else version
  ######################## paste this for gen_eval !!!!! ######################

  config.model_name = 'lsm_vit_mae'
  config.model = ml_collections.ConfigDict()
  config.model.patcher_config = Patcher_Config(
      hidden_size=384,
      kernel_size=(10, 2),
      groups=1,
      mode='2d',
  )
  config.model.num_heads = model_constants.NUM_HEADS[version]
  config.model.mlp_dim = model_constants.MLP_DIMS[version]
  config.model.num_layers = model_constants.NUM_LAYERS[version]
  config.model.dropout_rate = 0.0
  config.model.classifier = 'none'  # Has to be "none" for the autoencoder
  config.model.representation_size = None
  config.model.positional_embedding = 'sinusoidal_2d'
  config.model.positional_embedding_decoder = 'sinusoidal_2d'
  # decoder
  config.model.decoder_config = ml_collections.ConfigDict()
  config.model.decoder_config.hidden_size = (
      model_constants.DECODER_HIDDEN_SIZES[version]
  )
  config.model.decoder_config.mlp_dim = model_constants.DECODER_MLP_DIMS[
      version
  ]
  config.model.decoder_config.num_layers = model_constants.DECODER_NUM_LAYERS[
      version
  ]
  config.model.decoder_config.num_heads = model_constants.DECODER_NUM_HEADS[
      version
  ]
  config.model.decoder_config.dropout_rate = 0.0
  config.model.decoder_config.attention_dropout_rate = 0.0

  config.masked_feature_loss = ml_collections.ConfigDict()
  config.masked_feature_loss.targets_type = 'rgb'
  config.masked_feature_loss.loss_only_masked_tokens = LOSS_ONLY_MASKED_TOKENS
  config.masked_feature_loss.loss_type = 'squared'  # 'squared' or 'absolute'

  config.masker_config = Masker_Config(
      maskstrategy_list=[
          MaskStrategy_Config(
              strategy='random',
              mask_probability=0.8,
              weight=1,
              mask_dim='time',
          ),
      ],
      on_cpu=True,
      inherited=False,
  )

  # Datetime features.
  config.use_datetime_features = USE_DATETIME_FEATURES
  ######################## paste this for gen_eval !!!!! ######################

  # Training.
  config.trainer_name = 'lsm_mae_trainer'
  config.batch_size = 8 if runlocal else BATCH_SIZE
  config.num_training_steps = 100 if runlocal else NUM_TRAIN_STEPS
  config.log_eval_steps = LOG_EVAL_SUMMARY_STEPS
  config.log_summary_steps = LOG_TRAIN_SUMMARY_STEPS
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

  # Downstream Tasks.
  # TODO(girishvn): These (0, 1) need to be adapted to LSM datasets
  # 0) Linear Probing.
  # 1) Fewshot.

  # 2) Reconstruction Eval Tasks (Forecast and Imputation).
  config.forecast = LSM_PREDEFINED_CONFIGS['eval_fore_1day']
  config.imputation = LSM_PREDEFINED_CONFIGS['eval_imp_1day']
  config.random_imputation = LSM_PREDEFINED_CONFIGS['eval_randimp_1day']

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
      hyper.sweep('config.dataset_configs.train_num_samples', TRAIN_DATA_SIZES),
      hyper.sweep('config.use_train_augmentations', USE_TRAIN_AUGMENTATIONS),
      hyper.sweep('config.loss_ignore_imputation', LOSS_IGNORE_IMPUTATION),
      hyper.sweep('config.dataset_configs.dataset', TRAIN_DATASET_NAME),
  ])
