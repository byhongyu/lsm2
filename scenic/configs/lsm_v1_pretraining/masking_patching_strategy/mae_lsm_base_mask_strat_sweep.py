r"""A config to train a Base ViT MAE on LSM dataset.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
gxm third_party/py/scenic/google/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/masking_patching_strategy/mae_lsm_base_mask_strat_sweep.py \
--platform=vlp_4x8 \
--exp_name=lsm_mae_tier2_base_10_5_res_mask_strat_sweep \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_alloc=group:mobile-dynamic/h2o-ai-gqm-quota \
--scheduling_time_quantum=2d \
--priority=200

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/masking_patching_strategy/mae_lsm_base_mask_strat_sweep.py:runlocal
"""

import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants


# To set constants.
# 1) Dataset variables.
DATASET_NAME = 'lsm_300min_pretraining_165K_n10'
CACHE_DATASET = True
TRAIN_DATA_SIZES = [1321235]
USE_DATETIME_FEATURES = False
USE_TRAIN_AUGMENTATIONS = [True]
TRAIN_AUGMENTATIONS = ['stretch', 'flip', 'noise']
SHUFFLE_SEED = 42

# 2) Training / eval variables.
BATCH_SIZE = 4096
NUM_TRAIN_STEPS = 50000
LRS = [5e-3]
WEIGHT_DECAYS = [1e-4]

# 3) Logging variables.
LOG_EVAL_SUMMARY_STEPS = 500  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = 100  # LOG_EVAL_SUMMARY_STEPS * 5
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)

# Model variant / patch H (time steps) / patch W (features)
VARIANT = 'B/10/5'
TOKEN_MASK_PROB = [
    'constant_0.8_imputation_time',
    'constant_0.8_forecast_time',
    'constant_0.8_bar',
    'constant_0.8_bar_w',
]
LOSS_ONLY_MASKED_TOKENS = True

# Downstream Tasks.
# Imputation and forecast eval
RECONSTRUCTION_HORIZONS = [0.1, 0.2, 0.4]

# Linear probe eval.
LINEAR_PROBE_USE_TRAIN_AUGMENTATIONS = False
LINEAR_PROBE_TRAIN_AUGMENTATIONS = ['noise']


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)

  # Experiment.
  config = ml_collections.ConfigDict()
  config.experiment_name = f'electrodes-mae-{DATASET_NAME}'
  config.dataset_name = f'lsm_prod/{DATASET_NAME}'
  config.shuffle_seed = SHUFFLE_SEED

  # Dataset.
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = f'lsm_prod/{DATASET_NAME}'
  config.dataset_configs.num_classes = None
  config.dataset_configs.train_split = 'train'  # train data split
  config.dataset_configs.train_num_samples = TRAIN_DATA_SIZES[0]  # train sample
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

  version = 'Deb' if runlocal else version
  config.model_name = 'lsm_vit_mae'
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
  config.masked_feature_loss.token_mask_probability = TOKEN_MASK_PROB[0]
  config.masked_feature_loss.loss_only_masked_tokens = LOSS_ONLY_MASKED_TOKENS
  config.masked_feature_loss.loss_type = 'squared'  # 'squared' or 'absolute'

  # Datetime features.
  config.use_datetime_features = USE_DATETIME_FEATURES

  # Training.
  config.trainer_name = 'lsm_mae_trainer'
  config.batch_size = 8 if runlocal else BATCH_SIZE
  config.num_training_steps = NUM_TRAIN_STEPS
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

  # Downstream Tasks.
  # 2) Reconstruction Eval Tasks (Forecast and Imputation).
  config.forecast = ml_collections.ConfigDict()
  config.forecast.horizons = RECONSTRUCTION_HORIZONS
  config.imputation = ml_collections.ConfigDict()
  config.imputation.horizons = RECONSTRUCTION_HORIZONS

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
      hyper.sweep(
          'config.masked_feature_loss.token_mask_probability', TOKEN_MASK_PROB
      ),
      hyper.sweep('config.schedule.all.lr_configs.base_learning_rate', LRS),
      hyper.sweep('config.optimizer.weight_decay', WEIGHT_DECAYS),
      hyper.sweep('config.dataset_configs.train_num_samples', TRAIN_DATA_SIZES),
      hyper.sweep('config.use_train_augmentations', USE_TRAIN_AUGMENTATIONS),
  ])
