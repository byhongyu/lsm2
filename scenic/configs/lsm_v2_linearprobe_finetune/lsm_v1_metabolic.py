r"""A config to linear probe / finetune LSM V2 (ViT-S MAE) on metabolic data.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
xmanager launch experimental/largesensormodels/scenic/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/lsm_v2_linearprobe_finetune/lsm_v1_metabolic.py \
--platform=glp_4x8 \
--exp_name=lsm_v1_mae_small_metabolic_linearprobe \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_pool=plat-dev-dynamic \
--xm_resource_alloc=group:plat-dev-dynamic/h2o-ai-gqm-quota \
--xm_use_developer_builds_in_prod=false \
--cell=viglobal \
--priority=200

Note:
--priority=119 for Batch priority, --priority=25 for Freebie priority.
-- TPU usage is set for the following configurations:
    -- input of shape (MINUTES_IN_ONE_DAY, 26, 1)
    -- batch size of 1024
--glp_4x8 should work these configurations and lp / ft.
--viglobal is the cell for auto-selection.

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/lsm_v2_linearprobe_finetune/lsm_v1_metabolic.py:runlocal
"""

import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import patcher_config  # pylint: disable=unused-import
from google3.experimental.largesensormodels.scenic.trainers.masking import masker_config  # pylint: disable=unused-import
from google3.experimental.largesensormodels.scenic.utils import predefined_configs

# To set constants.
# 1) Dataset variables.
DATASET_NAME = 'metabolic_tfrecords_24h_missingness_80'
TASK_NAME = [
    'hypertension_binary',
    # 'homa_ir_binray',
    # 'hyperlipidemia_binary',
    # 'diabetes_binary',
    'anxiety_binary',
    # 'cardiovascular_binary',
    # 'respiratory',
]

CACHE_DATASET = True
TRAIN_DATA_SIZE = None
USE_TRAIN_AUGMENTATIONS = [False]
TRAIN_AUGMENTATIONS = ['noise']
USE_DATETIME_FEATURES = False
SHUFFLE_SEED = 42
SHUFFLE_BUFFER_SIZE = 10_000

# 2) Model variables.
MODEL_NAME = 'lsm_vit_mae'
MODEL_TRAINER = 'lsm_linear_probe_trainer'
LSM_PREDEFINED_CONFIGS = predefined_configs.LSM_PREDEFINED_CONFIGS
VARIANT = 'S'

# Checkpoint variables.
# XID/WID to load from.
XID = [159148907]  # LSM V1 Job (from Xin)
WID = [1, 2, 3, 4, 5]
WORK_DIR = '/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm'
INIT_CHECKPOINT_DIR = [f'{WORK_DIR}/{xid}/{wid}' for xid in XID for wid in WID]  # pylint: disable=g-complex-comprehension

# Specify a checkpoint step. (None defaults to the last checkpoint)
INIT_CHECKPOINT_STEP = [100_000]

PATCH_CONFIG = patcher_config.Patcher_Config(
    hidden_size=384,
    kernel_size=(10, 2),
    groups=1,
    mode='2d',
)

# Non-attention masking (Legacy.)
MASK_CONFIG = LSM_PREDEFINED_CONFIGS['downstream_noninherited_masking']
REPRESENTATION_POOLING_FN = ['mean']


# 3) Training variables.
# Loss variables.
LOSS_FN = ['balanced_softmax_loss']  # ['weighted_softmax_cross_entropy']
WEIGHTED_LOSS = [False]  # whether or not to use label weights
LABEL_SMOOTHING = [None]

# Training size.
BATCH_SIZE = 512
NUM_TRAIN_STEPS = 500

# Optimizer variables.
WEIGHT_DECAYS = [1e-4]

# 4) Logging variables.
LOG_EVAL_SUMMARY_STEPS = 25  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = 25  # LOG_EVAL_SUMMARY_STEPS * 5
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)

# Linear Probe Variables:
FINETUNE = [False]
LRS = [5e-3]
LINEAR_DROPOUT_RATE = [0.0]
# REPRESENTATION_POOLING_FN = ['mean']

# Finetune variables.
# TODO(girishvn): These have NOT been optimized.
# A comprehensive hparam sweep is needed.
# FINETUNE = [True]
# LRS = [5e-3]
# LINEAR_DROPOUT_RATE = [0.0]
# FINETUNE_WARMUP_STEPS = [0, NUM_TRAIN_STEPS // 5]
# REPRESENTATION_POOLING_FN = ['mean']

# Linear learned embedding variables.
# FINETUNE = [False]
# LRS = [5e-3, 5e-4, 5e-5]
# LINEAR_DROPOUT_RATE = [0.0]
# FINETUNE_WARMUP_STEPS = [0]
# REPRESENTATION_POOLING_FN = ['learned_mean']

# Conv pooling variables.
# FINETUNE = [False]
# LRS = [5e-3, 5e-4, 5e-5]
# LINEAR_DROPOUT_RATE = [0.0]
# FINETUNE_WARMUP_STEPS = [0]
# REPRESENTATION_POOLING_FN = ['temporal_conv']
# LINEAR_DROPOUT = [0.3]


# TODO(girishvn): MOVE TO SOMEWHERE BETTER
# Representation pooling variables.
REPRESENTATION_LAYER = ['pre_logits']

# Metadata encoding variables.
# METADATA_ENCODING_FN = ['identity', 'none']
METADATA_ENCODING_FN = ['none']


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)

  # 1. Experiment.
  config = ml_collections.ConfigDict()
  config.experiment_name = f'lsmv1_mae_linearprobe-{DATASET_NAME}'
  config.shuffle_seed = SHUFFLE_SEED

  # THINGS TO MOVE TO SOMEWHERE BETTER
  config.linear_probe_representation_layer = REPRESENTATION_LAYER[0]
  config.linear_finetune = FINETUNE[0]
  config.linear_probe_warmup_steps = None  # TODO(girishvn): unused.
  config.linear_dropout_rate = LINEAR_DROPOUT_RATE[0]
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_dir = INIT_CHECKPOINT_DIR[0]
  config.init_from.checkpoint_step = INIT_CHECKPOINT_STEP[0]
  config.linear_probe_metadata_method = METADATA_ENCODING_FN[0]

  # Representation Pooling.
  config.representation_pooling = ml_collections.ConfigDict()
  config.representation_pooling.method = REPRESENTATION_POOLING_FN[0]
  config.representation_pooling.time_window = (None, None)

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
  config.model_name = MODEL_NAME
  config.model = ml_collections.ConfigDict()

  # patcher
  config.model.patcher_config = PATCH_CONFIG

  # encoder
  config.model.num_heads = model_constants.NUM_HEADS[VARIANT]
  config.model.mlp_dim = model_constants.MLP_DIMS[VARIANT]
  config.model.num_layers = model_constants.NUM_LAYERS[VARIANT]
  config.model.dropout_rate = 0.0
  config.model.classifier = 'none'  # Has to be "none" for the autoencoder
  config.model.representation_size = None
  config.model.positional_embedding = 'sinusoidal_2d'
  config.model.positional_embedding_decoder = 'sinusoidal_2d'
  # decoder
  config.model.decoder_config = ml_collections.ConfigDict()
  config.model.decoder_config.hidden_size = (
      model_constants.DECODER_HIDDEN_SIZES[VARIANT]
  )
  config.model.decoder_config.mlp_dim = model_constants.DECODER_MLP_DIMS[
      VARIANT
  ]
  config.model.decoder_config.num_layers = model_constants.DECODER_NUM_LAYERS[
      VARIANT
  ]
  config.model.decoder_config.num_heads = model_constants.DECODER_NUM_HEADS[
      VARIANT
  ]
  config.model.decoder_config.dropout_rate = 0.0
  config.model.decoder_config.attention_dropout_rate = 0.0

  # 4. Model Loss
  config.classification_loss = ml_collections.ConfigDict()
  config.classification_loss.loss_name = LOSS_FN[0]
  config.classification_loss.label_smoothing = LABEL_SMOOTHING[0]
  config.classification_loss.weighted_loss = WEIGHTED_LOSS[0]

  config.masker_config = MASK_CONFIG

  # 5. Datetime features.
  config.use_datetime_features = USE_DATETIME_FEATURES

  # Training.
  config.trainer_name = MODEL_TRAINER
  config.batch_size = 8 if runlocal else BATCH_SIZE
  config.num_training_steps = 100 if runlocal else NUM_TRAIN_STEPS
  config.log_eval_steps = LOG_EVAL_SUMMARY_STEPS
  config.log_summary_steps = LOG_EVAL_SUMMARY_STEPS
  config.rng_seed = 42
  config.use_train_augmentations = USE_TRAIN_AUGMENTATIONS[0]
  config.train_augmentations = TRAIN_AUGMENTATIONS

  # TODO(girishvn): Check is LR scheduler is still useful for linear probe.
  # Learning rate schedule.
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

      # Init XID/WID checkpoint.
      hyper.sweep('config.init_from.checkpoint_dir', INIT_CHECKPOINT_DIR),
      hyper.sweep('config.init_from.checkpoint_step', INIT_CHECKPOINT_STEP),

      # Dataset / task parameters.
      hyper.sweep('config.dataset_configs.task_name', TASK_NAME),

      # Linear probe parameters.
      hyper.sweep('config.linear_finetune', FINETUNE),
      hyper.sweep('config.linear_probe_metadata_method', METADATA_ENCODING_FN),
      hyper.sweep(
          'config.representation_pooling.method', REPRESENTATION_POOLING_FN
      ),
      hyper.sweep(
          'config.linear_probe_representation_layer', REPRESENTATION_LAYER
      ),

      # Training hyperparameters.
      hyper.sweep('config.schedule.all.lr_configs.base_learning_rate', LRS),
  ])
