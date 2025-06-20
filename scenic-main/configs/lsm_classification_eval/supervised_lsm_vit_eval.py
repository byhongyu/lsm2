r"""A config to train a TinyShallow ViT MAE on LSM dataset.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
gxm third_party/py/scenic/google/xm/launch_xm.py -- \
--binary //experimental/largesensormodels/scenic:main \
--config=experimental/largesensormodels/scenic/configs/lsm_classification_eval/supervised_lsm_vit_eval.py \
--platform=vlp_2x2 \
--exp_name=lsm_supervised_tier2_vit_base_eval \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_alloc=group:mobile-dynamic/h2o-ai-gqm-quota \
--xm_use_developer_builds_in_prod=false \
--priority=200

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//experimental/largesensormodels/scenic:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/lsm_classification_eval/supervised_lsm_vit_eval.py:runlocal
"""


from typing import Optional  # pylint: disable=unused-import
import ml_collections
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants  # pylint: disable=unused-import


# To set constants.
# XM Job to Init Model From.
INIT_CHECKPOINT_DIR = [
    # ACTIVITY RECOGNITION
    # Supervised fewshot
    # '/cns/dz-d/home/xliucs/lsm/xm/127069108/3',  # 20 spc
    # '/cns/dz-d/home/xliucs/lsm/xm/127068936/3',  # 15 spc
    # '/cns/dz-d/home/xliucs/lsm/xm/127068506/3',  # 10 spc
    # '/cns/dz-d/home/xliucs/lsm/xm/127063814/3',  # 5 spc

    # Baseline model
    # '/cns/dz-d/home/xliucs/lsm/xm/126653662/2'  # Full dataset train

    # MOOD VS ACTIVITY
    # Baseline model
    '/cns/dz-d/home/xliucs/lsm/xm/127354350/3'  # Full dataset train
]
# Specify a checkpoint step. (None defaults to the last check)
INIT_CHECKPOINT_STEP = None


# 1) Dataset variables.
DATASET_NAME = 'lsm_300min_mood_vs_activity'
NUM_TRAIN_STEPS = 400

# DATASET_NAME = 'lsm_300min_600_activities_remapped_8class'
# NUM_TRAIN_STEPS = 300

CACHE_DATASET = True
TRAIN_DATA_SIZE = None
USE_DATETIME_FEATURES = False
USE_TRAIN_AUGMENTATIONS = [True]
TRAIN_AUGMENTATIONS = ['noise']
SHUFFLE_SEED = 42

WEIGHTED_LOSS = [False]  # whether or not to use label weights
LABEL_SMOOTHING = [None]

# 2) Training / eval variables.
BATCH_SIZE = 128
LRS = [5e-4]
WEIGHT_DECAYS = [1e-4]

# 3) Logging variables.
LOG_EVAL_SUMMARY_STEPS = 25  # STEPS_PER_EPOCH
LOG_CHECKPOINT_STEPS = 25  # LOG_EVAL_SUMMARY_STEPS * 5
MAX_NUM_CHECKPOINTS = int(NUM_TRAIN_STEPS / LOG_CHECKPOINT_STEPS)

# Model variant
VARIANT = 'B/10/5'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration."""

  runlocal = bool(runlocal)

  # Experiment.
  config = ml_collections.ConfigDict()
  config.experiment_name = f'electrodes-supervised-{DATASET_NAME}_eval'
  config.dataset_name = f'lsm_prod/{DATASET_NAME}'
  config.shuffle_seed = SHUFFLE_SEED

  config.init_from = ml_collections.ConfigDict()
  config.init_from.checkpoint_dir = INIT_CHECKPOINT_DIR[0]
  config.init_from.checkpoint_step = INIT_CHECKPOINT_STEP

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
  config.model_name = 'vit'
  config.model = ml_collections.ConfigDict()
  config.model.mlp_dim = model_constants.MLP_DIMS[version]
  config.model.num_layers = model_constants.NUM_LAYERS[version]
  config.model.num_heads = model_constants.NUM_HEADS[version]
  config.model.positional_embedding = 'learned_1d'
  config.model.representation_size = None
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = tuple([int(patch_h), int(patch_w)])
  config.model.hidden_size = model_constants.HIDDEN_SIZES[version]
  config.model.classifier = 'gap'
  config.model.dropout_rate = 0.0
  config.model.attention_dropout_rate = 0.0

  # Datetime features.
  config.use_datetime_features = USE_DATETIME_FEATURES

  # Training.
  config.trainer_name = 'lsm_supervised_evaluator'
  config.batch_size = 8 if runlocal else BATCH_SIZE
  config.num_training_steps = NUM_TRAIN_STEPS
  config.log_eval_steps = LOG_EVAL_SUMMARY_STEPS
  config.log_summary_steps = LOG_EVAL_SUMMARY_STEPS
  config.rng_seed = 42
  config.use_train_augmentations = USE_TRAIN_AUGMENTATIONS[0]
  config.train_augmentations = TRAIN_AUGMENTATIONS

  config.weighted_loss = WEIGHTED_LOSS[0]
  config.label_smoothing = LABEL_SMOOTHING[0]

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
      hyper.sweep('config.init_from.checkpoint_dir', INIT_CHECKPOINT_DIR),
      hyper.sweep('config.label_smoothing', LABEL_SMOOTHING),
      hyper.sweep('config.use_train_augmentations', USE_TRAIN_AUGMENTATIONS),
      hyper.sweep('config.weighted_loss', WEIGHTED_LOSS),
      hyper.sweep('config.schedule.all.lr_configs.base_learning_rate', LRS),
      hyper.sweep('config.optimizer.weight_decay', WEIGHT_DECAYS),
  ])
