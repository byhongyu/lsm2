# pylint: disable=line-too-long
r"""A config to train a very small ViT MAE on CIFAR-10.

Forked from google3/third_party/py/scenic/projects/multimask/configs/mae_cifar10_tiny.py

To run on XManager:
gxm third_party/py/scenic/google/xm/launch_xm.py -- \
--binary //third_party/py/scenic/projects/multimask:main \
--config=experimental/largesensormodels/scenic/configs/mae_cifar10_tiny.py \
--platform=jd_2x2 \
--exp_name=mm_mae_cifar10_tishallow16_64res_test1 \
--workdir=/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/xm/\{xid\} \
--xm_resource_alloc=group:cml/cml-shared-ml-user \
--priority=200

To run locally:
./third_party/py/scenic/google/runlocal.sh \
--uptc="" \
--binary=//third_party/py/scenic/projects/multimask:main \
--config=$(pwd)/experimental/largesensormodels/scenic/configs/mae_cifar10_tiny.py:runlocal
"""
# pylint: disable=line-too-long

import ml_collections
from scenic.projects.baselines.configs.google.common import common_fewshot

NUM_TRAIN_STEPS = 5000
VARIANT = 'TiShallow/8'
LRS = [1e-3]
TOKEN_MASK_PROB = 'constant_0.8'

HIDDEN_SIZES = {
    'Deb': 16,
    'Ti': 192,
    'TiShallow': 192,
    'S': 384,
    'SShallow': 384,
    'M': 512,
    'B': 768,
    'L': 1024,
    'H': 1280,
    'g': 1408,
    'G': 1664,
    'e': 1792,
}
MLP_DIMS = {
    'Deb': 32,
    'Ti': 768,
    'TiShallow': 768,
    'S': 1536,
    'SShallow': 1536,
    'M': 2048,
    'B': 3072,
    'L': 4096,
    'H': 5120,
    'g': 6144,
    'G': 8192,
    'e': 15360,
}
NUM_HEADS = {
    'Deb': 2,
    'Ti': 3,
    'TiShallow': 3,
    'S': 6,
    'SShallow': 6,
    'M': 8,
    'B': 12,
    'L': 16,
    'H': 16,
    'g': 16,
    'G': 16,
    'e': 16,
}
NUM_LAYERS = {
    'Deb': 2,
    'Ti': 12,
    'TiShallow': 4,
    'S': 12,
    'SShallow': 4,
    'M': 12,
    'B': 12,
    'L': 24,
    'H': 32,
    'g': 40,
    'G': 48,
    'e': 56,
}


DECODER_HIDDEN_SIZES = {
    'Deb': 16,
    'Ti': 128,
    'TiShallow': 128,
    'S': 256,
    'B': 512,
    'L': 512,
    'H': 512,
}
DECODER_MLP_DIMS = {
    'Deb': 32,
    'Ti': 512,
    'TiShallow': 512,
    'S': 1024,
    'B': 2048,
    'L': 2048,
    'H': 2048,
}
DECODER_NUM_LAYERS = {
    'Deb': 2,
    'Ti': 2,
    'TiShallow': 2,
    'S': 4,
    'B': 8,
    'L': 8,
    'H': 8,
}
DECODER_NUM_HEADS = {
    'Deb': 2,
    'Ti': 4,
    'TiShallow': 4,
    'S': 8,
    'B': 16,
    'L': 16,
    'H': 16,
}


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'cifar10-mae-vit-tiny'
  # Dataset.
  config.dataset_name = 'bit'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'cifar10'
  # config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'test'
  config.dataset_configs.pp_train = (
      'decode'
      '|resize_small(80)|random_crop(64)|flip_lr'
      '|value_range(-1, 1)'
      '|copy("label", "labels")'
  )
  config.dataset_configs.pp_eval = (
      'decode'
      '|resize_small(80)|central_crop(64)'
      '|value_range(-1, 1)'
      '|copy("label", "labels")'
  )
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  version, patch = VARIANT.split('/')
  version = 'Deb' if runlocal else version
  config.model_name = 'vit_mae'
  config.model = ml_collections.ConfigDict()
  # encoder
  config.model.hidden_size = HIDDEN_SIZES[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = NUM_HEADS[version]
  config.model.mlp_dim = MLP_DIMS[version]
  config.model.num_layers = NUM_LAYERS[version]
  config.model.dropout_rate = 0.0
  config.model.classifier = 'none'  # Has to be "none" for the autoencoder
  config.model.representation_size = None
  config.model.positional_embedding = 'sinusoidal_2d'
  config.model.positional_embedding_decoder = 'sinusoidal_2d'
  # decoder
  config.model.decoder_config = ml_collections.ConfigDict()
  config.model.decoder_config.hidden_size = DECODER_HIDDEN_SIZES[version]
  config.model.decoder_config.mlp_dim = DECODER_MLP_DIMS[version]
  config.model.decoder_config.num_layers = DECODER_NUM_LAYERS[version]
  config.model.decoder_config.num_heads = DECODER_NUM_HEADS[version]
  config.model.decoder_config.dropout_rate = 0.0
  config.model.decoder_config.attention_dropout_rate = 0.0

  config.masked_feature_loss = ml_collections.ConfigDict()
  config.masked_feature_loss.targets_type = 'rgb'
  config.masked_feature_loss.token_mask_probability = TOKEN_MASK_PROB
  config.masked_feature_loss.loss_only_masked_tokens = True
  config.masked_feature_loss.loss_type = 'squared'  # 'squared' or 'absolute'

  # Training.
  config.trainer_name = 'multimask_trainer'
  config.batch_size = 8 if runlocal else 1024
  config.num_training_steps = NUM_TRAIN_STEPS
  config.log_eval_steps = 500
  config.log_summary_steps = 100
  config.rng_seed = 42
  sched = ml_collections.ConfigDict()
  sched.re = '(.*)'
  sched.lr_configs = ml_collections.ConfigDict()
  sched.lr_configs.learning_rate_schedule = 'compound'
  sched.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  sched.lr_configs.total_steps = NUM_TRAIN_STEPS
  sched.lr_configs.steps_per_cycle = sched.lr_configs.total_steps
  sched.lr_configs.warmup_steps = 2000
  sched.lr_configs.base_learning_rate = LRS[0]
  config.schedule = ml_collections.ConfigDict({'all': sched})

  # *Single* optimizer.
  optim = ml_collections.ConfigDict()
  optim.optax_name = 'scale_by_adam'
  # optim.optax = dict(mu_dtype='bfloat16')
  optim.optax_configs = ml_collections.ConfigDict({  # Optimizer settings.
      'b1': 0.9,
      'b2': 0.999,
  })
  config.optax = dict(mu_dtype='bfloat16')
  optim.max_grad_norm = 1.0

  optim.weight_decay = 1e-4
  optim.weight_decay_decouple = True
  config.optimizer = optim

  # Fewshot.
  config.fewshot = common_fewshot.get_config(
      batch_size=config.batch_size, target_resolution=64, resize_resolution=80
  )
  config.fewshot.datasets = {
      'cifar100': ('cifar100', 'train', 'test'),
      'cifar10': ('cifar10', 'train', 'test'),
  }
  config.fewshot.walk_first = ('cifar10', 10)
  config.fewshot.representation_layer = 'pre_logits'
  config.fewshot.log_eval_steps = 1000

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 1000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

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
  ])


# END GOOGLE-INTERNAL
