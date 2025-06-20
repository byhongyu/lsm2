"""Common config for linear probe."""

from typing import List, Optional
import ml_collections


def get_single_dataset_linear_probe_config(
    *,
    dataset_name: str,
    log_eval_steps: int,
    model_config: ml_collections.ConfigDict,
    use_datetime_features: bool = False,
    use_train_augmentations: bool = True,
    train_augmentations: Optional[List[str]] = None,
    cache_dataset: bool = True,
    runlocal: bool = False,
    masked_feature_loss: bool = False,
) -> ml_collections.ConfigDict:
  """Returns a standard-ish fewshot eval configuration."""
  linear_probe_config = ml_collections.ConfigDict()
  linear_probe_config.walk_first = None
  linear_probe_config.data_dtype_str = 'float32'
  linear_probe_config.representation_layer = 'pre_logits'
  linear_probe_config.log_eval_steps = log_eval_steps
  linear_probe_config.log_train_summary_steps = log_eval_steps
  linear_probe_config.dataset_configs = ml_collections.ConfigDict()
  linear_probe_config.dataset_configs.dataset = f'lsm_prod/{dataset_name}'
  linear_probe_config.dataset_configs.train_split = 'train'  # train data split
  linear_probe_config.dataset_configs.train_num_samples = None  # all samples
  # eval data split - note: this split is used for validation and test.
  linear_probe_config.dataset_configs.eval_split = (
      'test[:16]' if runlocal else 'test'
  )
  linear_probe_config.gather_to_host = False if runlocal else False
  linear_probe_config.dataset_configs.cache_dataset = cache_dataset
  linear_probe_config.dataset_configs.prefetch_to_device = 2
  linear_probe_config.dataset_configs.shuffle_buffer_size = 250_000
  linear_probe_config.model = model_config
  linear_probe_config.use_train_augmentations = use_train_augmentations
  linear_probe_config.train_augmentations = train_augmentations
  linear_probe_config.use_datetime_features = use_datetime_features
  linear_probe_config.masked_feature_loss = masked_feature_loss
  # Eg: google3/third_party/py/scenic/projects/mae/configs/common/linearprobe.py
  linear_probe_config.lr_configs = ml_collections.ConfigDict()
  linear_probe_config.lr_configs.learning_rate_schedule = 'compound'
  linear_probe_config.lr_configs.factors = (
      'constant * cosine_decay * linear_warmup'
  )
  linear_probe_config.lr_configs.total_steps = 400
  linear_probe_config.lr_configs.steps_per_cycle = (
      linear_probe_config.lr_configs.total_steps
  )
  linear_probe_config.lr_configs.warmup_steps = int(
      linear_probe_config.lr_configs.total_steps * 0.05
  )
  linear_probe_config.lr_configs.base_learning_rate = 5e-3
  linear_probe_config.optimizer = 'adamw'
  linear_probe_config.weight_decay = 1e-4
  linear_probe_config.optax = dict(mu_dtype='bfloat16')
  linear_probe_config.num_training_epochs = 10
  linear_probe_config.label_smoothing = None
  linear_probe_config.batch_size = 256
  return linear_probe_config


def get_linear_probe_config(
    log_eval_steps: int,
    model_config: ml_collections.ConfigDict,
    use_datetime_features: bool = False,
    use_train_augmentations: bool = True,
    train_augmentations: Optional[List[str]] = None,
    cache_dataset: bool = True,
    runlocal: bool = False,
    masked_feature_loss: bool = False,
):

  return {
      # # Mood classification.
      # 'lsm_300min_2000_mood_balanced': get_single_dataset_linear_probe_config(
      #     dataset_name='lsm_300min_2000_mood_balanced',
      #     log_eval_steps=log_eval_steps,
      #     model_config=model_config,
      #     use_datetime_features=use_datetime_features,
      #     use_train_augmentations=use_train_augmentations,
      #     train_augmentations=train_augmentations,
      #     cache_dataset=cache_dataset,
      #     runlocal=runlocal,
      #     masked_feature_loss=masked_feature_loss,
      # ),

      # # Binary stress classification.
      # 'lsm_300min_2000_stress_balanced': get_single_dataset_linear_probe_config(
      #     dataset_name='lsm_300min_2000_stress_balanced',
      #     log_eval_steps=log_eval_steps,
      #     model_config=model_config,
      #     use_datetime_features=use_datetime_features,
      #     use_train_augmentations=use_train_augmentations,
      #     train_augmentations=train_augmentations,
      #     cache_dataset=cache_dataset,
      #     runlocal=runlocal,
      #     masked_feature_loss=masked_feature_loss,
      # ),

      # # Activity classification (10).
      # 'lsm_300min_600_activities_balanced_large': (
      #     get_single_dataset_linear_probe_config(
      #         dataset_name='lsm_300min_600_activities_balanced_v4',
      #         log_eval_steps=log_eval_steps,
      #         model_config=model_config,
      #         use_datetime_features=use_datetime_features,
      #         use_train_augmentations=use_train_augmentations,
      #         train_augmentations=train_augmentations,
      #         cache_dataset=cache_dataset,
      #         runlocal=runlocal,
      #         masked_feature_loss=masked_feature_loss,
      #     )
      # ),

      # # Mood vs activity differentiation (simple binary task).
      # 'lsm_300min_mood_vs_activity': get_single_dataset_linear_probe_config(
      #     dataset_name='lsm_300min_mood_vs_activity',
      #     log_eval_steps=log_eval_steps,
      #     model_config=model_config,
      #     use_datetime_features=use_datetime_features,
      #     use_train_augmentations=use_train_augmentations,
      #     train_augmentations=train_augmentations,
      #     cache_dataset=cache_dataset,
      #     runlocal=runlocal,
      #     masked_feature_loss=masked_feature_loss,
      # ),
  }
