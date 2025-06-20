r"""A config to train a Small ViT MAE on LSM dataset.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
gxm third_party/py/scenic/google/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/mae_lsm_tiny_linear_probe.py \
--platform=vlp_4x4 \
--exp_name=lsm_mae_tiny_10x5res_linear_probe \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_alloc=group:mobile-dynamic/h2o-ai-gqm-quota \
--xm_use_developer_builds_in_prod=false \
--scheduling_time_quantum=5h \
--priority=200

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/mae_lsm_tiny_linear_probe.py:runlocal
"""

import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants


# To set constants.
# XM Job to Init Model From.
XID = 124248449
INIT_CHECKPOINT_DIR = [
    f'/cns/dz-d/home/xliucs/lsm/xm/{XID}/1',
    f'/cns/dz-d/home/xliucs/lsm/xm/{XID}/2',
    f'/cns/dz-d/home/xliucs/lsm/xm/{XID}/3',
    f'/cns/dz-d/home/xliucs/lsm/xm/{XID}/4',
    f'/cns/dz-d/home/xliucs/lsm/xm/{XID}/5',
]
# Specify a checkpoint step. (None defaults to the last check)
INIT_CHECKPOINT_STEP = [101, 10001, 20001, 30001, 40001, 50000]

# 1) Dataset variables.
# DATASET_NAME = 'lsm_300min_mood_vs_activity'
# DATASET_NAME = 'lsm_300min_600_activities_balanced_v4'
DATASET_NAME = 'lsm_300min_600_activities_remapped_8class'
CACHE_DATASET = True
USE_DATETIME_FEATURES = False
USE_TRAIN_AUGMENTATIONS = [True]
TRAIN_AUGMENTATIONS = ['noise']
SHUFFLE_SEED = 42

# 2) Training / eval variables.
BATCH_SIZES = [128]
NUM_TRAIN_STEPS = 500


# LINEAR PROBE PARAMETERS.
# LRS = [5e-1]
# WEIGHT_DECAYS = [1e-4]
# FINETUNE = [False]
# REPRESENTATION_POOLING_FN = ['mean']
# LINEAR_DROPOUT = [0.3]

# CONVOLUTIONAL PROBE PARAMETERS.
LRS = [5e-3]
WEIGHT_DECAYS = [1e-4]
FINETUNE = [False]
REPRESENTATION_POOLING_FN = ['temporal_conv']

# FINE_TUNE PARAMETERS.
# LRS = [5e-4]
# WEIGHT_DECAYS = [1e-4]
# FINETUNE = [True]
# REPRESENTATION_POOLING_FN = ['mean']
# LINEAR_DROPOUT = [0.7]


# Other classification parameters.
WEIGHTED_LOSS = [False]  # whether or not to use label weights
LABEL_SMOOTHING = None

# 3) Logging variables.
LOG_EVAL_SUMMARY_STEPS = 25  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = 25  # LOG_EVAL_SUMMARY_STEPS * 5
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)

# Model variant / patch H (time steps) / patch W (features)
VARIANT = 'TiShallow/10/5'
TOKEN_MASK_PROB = 'constant_0.8'
LOSS_ONLY_MASKED_TOKENS = True

# Downstream Tasks.
REPRESENTATION_LAYER = 'pre_logits'
REPRESENTAION_POOLING_WINDOW = [(0.0, 1.0)]


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)

  # Experiment.
  config = ml_collections.ConfigDict()
  config.experiment_name = f'electrodes-lsm_lp-{DATASET_NAME}'
  config.dataset_name = f'lsm_prod/{DATASET_NAME}'
  config.shuffle_seed = SHUFFLE_SEED

  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_dir = INIT_CHECKPOINT_DIR[0]
  config.init_from.checkpoint_step = INIT_CHECKPOINT_STEP[0]

  # Dataset.
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = f'lsm_prod/{DATASET_NAME}'
  config.dataset_configs.num_classes = None
  config.dataset_configs.train_split = 'train'  # train data split
  config.dataset_configs.train_num_samples = None
  # eval data split - note: this split is used for validation and test.
  config.dataset_configs.eval_split = 'test[:64]' if runlocal else 'test'
  config.dataset_configs.cache_dataset = CACHE_DATASET
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  if len(VARIANT.split('/')) == 3:
    version = VARIANT.split('/')[0]  # model variant
    patch_h = VARIANT.split('/')[1]  # patch width
    patch_w = VARIANT.split('/')[2]  # patch height
  elif len(VARIANT.split('/')) == 2:
    version = VARIANT.split('/')[0]  # model variant
    patch_h = VARIANT.split('/')[1]  # patch width
    patch_w = VARIANT.split('/')[1]  # patch height
  else:
    raise ValueError(f'Invalid model variant: {VARIANT}')

  config.model_name = 'lsm_vit_mae'
  config.linear_probe_representation_layer = REPRESENTATION_LAYER
  config.linear_finetune = FINETUNE[0]
  config.linear_dropout_rate = 0.0

  # Representation Pooling.
  config.representation_pooling = ml_collections.ConfigDict()
  config.representation_pooling.method = REPRESENTATION_POOLING_FN[0]
  config.representation_pooling.time_window = (None, None)

  config.model = ml_collections.ConfigDict()
  # encoder
  config.model.hidden_size = model_constants.HIDDEN_SIZES[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = tuple([int(patch_h), int(patch_w)])
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
  config.masked_feature_loss.token_mask_probability = TOKEN_MASK_PROB
  config.masked_feature_loss.loss_only_masked_tokens = LOSS_ONLY_MASKED_TOKENS
  config.masked_feature_loss.loss_type = 'squared'  # 'squared' or 'absolute'

  # Datetime features.
  config.use_datetime_features = USE_DATETIME_FEATURES

  # Training.
  config.trainer_name = 'lsm_linear_probe_trainer'
  config.batch_size = 9 if runlocal else BATCH_SIZES[0]
  config.num_training_steps = NUM_TRAIN_STEPS
  config.log_eval_steps = LOG_EVAL_SUMMARY_STEPS
  config.log_summary_steps = LOG_EVAL_SUMMARY_STEPS
  config.rng_seed = 42
  config.use_train_augmentations = USE_TRAIN_AUGMENTATIONS[0]
  config.train_augmentations = TRAIN_AUGMENTATIONS

  config.weighted_loss = WEIGHTED_LOSS[0]
  config.label_smoothing = LABEL_SMOOTHING

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
  config.label_smoothing = None

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
      # Set parameters.
      hyper.sweep('config.use_train_augmentations', USE_TRAIN_AUGMENTATIONS),
      hyper.sweep('config.weighted_loss', WEIGHTED_LOSS),
      hyper.sweep('config.batch_size', BATCH_SIZES),
      hyper.sweep('config.schedule.all.lr_configs.base_learning_rate', LRS),
      hyper.sweep('config.optimizer.weight_decay', WEIGHT_DECAYS),
      # Swept parameters.
      hyper.sweep('config.linear_finetune', FINETUNE),
      hyper.sweep(
          'config.representation_pooling.method', REPRESENTATION_POOLING_FN
      ),
      # hyper.sweep('config.linear_dropout_rate', LINEAR_DROPOUT),
      hyper.sweep('config.init_from.checkpoint_dir', INIT_CHECKPOINT_DIR),
      hyper.sweep('config.init_from.checkpoint_step', INIT_CHECKPOINT_STEP),
  ])
