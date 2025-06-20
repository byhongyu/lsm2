r"""A config to run geneative evaluation a Base ViT MAE on LSM dataset.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
gxm experimental/largesensormodels/scenic/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/lsm_v2_embedding_dump/lsm_v2_mae_embedding_dump.py \
--platform=glp_2x4 \
--exp_name=lsm_v2_metabolic_embedding_dump \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_alloc=group:plat-dev-dynamic/h2o-ai-gqm-quota \
--xm_use_developer_builds_in_prod=false \
--cell=viglobal \
--priority=200

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/lsm_v2_embedding_dump/lsm_v2_mae_embedding_dump.py:runlocal
"""

import datetime
import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants
from google3.experimental.largesensormodels.scenic.utils.predefined_configs import LSM_PREDEFINED_CONFIGS

DATASET_NAME = 'metabolic_tfrecords_daily_alllabels_v05'
OUTPUT_DIR = '/namespace/fitbit-medical-sandboxes/partner/encrypted/chr-ards-metabolichealth/deid/exp/aliheydari/metabolic_embedding_dump'
# To set constants.
INIT_CHECKPOINT_DIR = [
    '/cns/dz-d/home/xliucs/lsm/xm/149985992/1'
]

INIT_CHECKPOINT_STEP = [50000]
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
# 1) Dataset variables.
CACHE_DATASET = False
USE_DATETIME_FEATURES = False
USE_TRAIN_AUGMENTATIONS = [False]
TRAIN_AUGMENTATIONS = []  # Do we need to set this?
SHUFFLE_SEED = 42
SHUFFLE_BUFFER_SIZE = None

# 2) Training / eval variables.
BATCH_SIZE = 64
# Model variant / patch H (time steps) / patch W (features)
VARIANT = 'B'


################ Unused Config Vars. ####################

NUM_TRAIN_STEPS = 50000
LRS = [5e-3]
WEIGHT_DECAYS = [1e-4]

# 3) Logging variables.
LOG_EVAL_SUMMARY_STEPS = 500  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = 100  # LOG_EVAL_SUMMARY_STEPS * 5
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)

TOKEN_MASK_PROB = 'constant_0.8'  # This is not being used.
LOSS_ONLY_MASKED_TOKENS = True


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)
  # Experiment.
  config = ml_collections.ConfigDict()
  config.dataset_name = DATASET_NAME
  config.shuffle_seed = SHUFFLE_SEED
  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_dir = INIT_CHECKPOINT_DIR[0]
  config.init_from.checkpoint_step = INIT_CHECKPOINT_STEP[0]
  config.init_from.timestamp = TIMESTAMP
  # Dataset.
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = DATASET_NAME
  if runlocal:
    config.output_dir = '/tmp'
  else:
    config.output_dir = OUTPUT_DIR
  config.dataset_configs.num_classes = None
  config.dataset_configs.train_split = 'train'  # train data split
  config.dataset_configs.train_num_samples = None  # train sample
  config.dataset_configs.eval_split = 'valid'
  config.dataset_configs.eval_num_samples = None
  config.dataset_configs.cache_dataset = CACHE_DATASET
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = SHUFFLE_BUFFER_SIZE

  # Embedding dump.
  config.embedding_dump = ml_collections.ConfigDict()
  config.embedding_dump.representation_layer = 'pre_logits'
  config.embedding_dump.splits_to_dump = ['train', 'valid', 'test']

  # Model.
  version = VARIANT

  version = 'Deb' if runlocal else version
  config.model_name = 'lsm_vit_mae'
  config.model = ml_collections.ConfigDict()
  # encoder

  # patcher
  config.model.patcher_config = LSM_PREDEFINED_CONFIGS[
      f'{version}__patcher_config__10by1_sharedpatch'
  ]

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
  config.trainer_name = 'lsm_embedding_dump'
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
      hyper.sweep('config.init_from.checkpoint_dir', INIT_CHECKPOINT_DIR),
      hyper.sweep('config.init_from.checkpoint_step', INIT_CHECKPOINT_STEP),
      hyper.sweep('config.schedule.all.lr_configs.base_learning_rate', LRS),
      hyper.sweep('config.optimizer.weight_decay', WEIGHT_DECAYS),
      hyper.sweep('config.use_train_augmentations', USE_TRAIN_AUGMENTATIONS),
  ])
