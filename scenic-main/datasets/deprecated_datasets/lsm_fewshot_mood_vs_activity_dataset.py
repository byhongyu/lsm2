"""Electrodes dataset data preprocesser and loader.

This dataset combines the activity (600) and mood (2000) datasets into a single
dual-label classification task (0 - mood evet, 1 - activity event).
This is a HACKY implementation for early testing.

Adapted from a combination of the following files:
google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py
google3/third_party/py/scenic/dataset_lib/dataset_utils.py
"""

import functools
from typing import Optional

from absl import logging
import jax.numpy as jnp
import jax.profiler
import ml_collections  # pylint: disable=unused-import
from scenic.dataset_lib import dataset_utils
import tensorflow as tf
import tensorflow_datasets as tfds

from google3.experimental.largesensormodels.scenic.datasets import dataset_constants
from google3.experimental.largesensormodels.scenic.datasets import dataset_utils as lsm_dataset_utils
from google3.experimental.largesensormodels.scenic.datasets import lsm_mood_vs_activity_dataset


def update_metadata(metadata, dataset_name, patch_size, samples_per_class):
  """Update metadata to reflect resizing and addition of datetime features."""
  # Setup: Get dataset name, feature shape, and possible datetime features.
  metadata_update = dict()
  dataset_name = dataset_name.split('/')[-1]
  time_features = dataset_constants.lsm_dataset_constants[dataset_name].get(
      'datetime_features', None
  )
  feature_shape = list(metadata['input_shape'][1:])
  feature_indices = list(range(feature_shape[1]))

  # Split features from time series features
  # NOTE: This assumes that the original 'input_signal' field has sensor
  # features contactanated to datetime features along the feature (w) dimension.
  if time_features is not None:
    # Get datetime indicies
    time_feature_indices = list(time_features['indices'])
    # Remove datetime indices from feature indices
    feature_indices = list(set(feature_indices) - set(time_features['indices']))
    # Get updated feature and datetime feature shapes.
    time_feature_shape = feature_shape.copy()  # update time feature shape
    time_feature_shape[1] = len(time_feature_indices)
    feature_shape[1] = len(feature_indices)  # update feature shape
  else:
    time_feature_shape = None

  # Padding: Update shape to reflect padding (for perfect patching).
  # valid_feats arrays denote which features are valid (1) vs padded (0).
  # 1. Update for sensor features
  _, pad_w, feat_shape_new = lsm_dataset_utils.get_height_crop_width_pad(
      tuple(feature_shape), patch_size
  )
  valid_feat_mask = [0] * pad_w[0] + [1] * feature_shape[1] + [0] * pad_w[1]
  metadata_update['input_shape'] = tuple([-1] + list(feat_shape_new))
  metadata_update['input_valid_feats'] = tuple(valid_feat_mask)

  # 2. Update for datetime features
  if time_features is not None:
    _, time_pad_w, time_feature_shape_new = (
        lsm_dataset_utils.get_height_crop_width_pad(
            tuple(time_feature_shape), patch_size
        )
    )
    valid_time_feat_mask = (
        [0] * time_pad_w[0] + [1] * time_feature_shape[1] + [0] * time_pad_w[1]
    )
    metadata_update['datetime_input_shape'] = tuple(
        [-1] + list(time_feature_shape_new)
    )
    metadata_update['datime_valid_feats'] = tuple(valid_time_feat_mask)

  else:
    metadata_update['datetime_input_shape'] = None
    metadata_update['datime_valid_feats'] = None

  # 6. Add label counts.
  label_counts = metadata['label_counts']
  label_counts = [
      int(tf.math.minimum(lc, samples_per_class)) for lc in label_counts
  ]
  metadata_update['label_counts'] = label_counts

  return metadata_update


def fewshot_filter_examples(ds, n):
  """Take n samples for each allowed class."""

  def filter_by_class(class_label):
    if class_label == 'mood':
      return lambda x: tf.cast(x['metadata']['mood_log'], tf.bool)
    elif class_label == 'activity':
      return lambda x: tf.cast(x['metadata']['exercise_log'], tf.bool)
    else:
      raise ValueError(f'Class label {class_label} is not supported.')

  # Create empty list to hold datasets for each class and loop through each
  # class and filter samples.
  filtered_datasets = []
  for label in ['mood', 'activity']:
    class_samples = ds.filter(filter_by_class(label)).take(n)
    filtered_datasets.append(class_samples)

  # Concat class datasets.
  fewshot_dataset = filtered_datasets[0]
  for ds in filtered_datasets[1:]:
    fewshot_dataset = fewshot_dataset.concatenate(ds)

  return fewshot_dataset


def get_electrodes_mood_vs_activity_dataset(
    *,
    config,
    num_shards,
    batch_size,
    eval_batch_size=None,
    dtype_str='float32',
    shuffle_seed=0,
    rng=None,
    shuffle_buffer_size=None,
    dataset_service_address: Optional[str] = None,
    dataset_name: Optional[str] = None,  # pylint: disable=unused-argument
    data_dir='/namespace/fitbit-medical-sandboxes/partner/encrypted/chr-ards-electrodes/deid/exp/dmcduff/ttl=6w/msa_1_5/lsm_tfds_datasets',
):
  """Gets and formats the Electrodes dataset.

  Adapted from:
  google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py and
  google3/third_party/py/scenic/dataset_lib/dataset_utils.py.

  Args:
    config: ml_collections.ConfigDict; Config for the experiment.
    num_shards: int; Number of shards to split the dataset into.
    batch_size: int; Batch size for training.
    eval_batch_size: int; Batch size for evaluation.
    dtype_str: str; Data type of the image.
    shuffle_seed: int; Seed for shuffling the dataset.
    rng: jax.random.PRNGKey; Random number generator key.
    shuffle_buffer_size: int; Size of the shuffle buffer.
    dataset_service_address: str; Address of the dataset service.
    dataset_name: str; Name of the dataset.
    data_dir: str; Directory of the dataset.

  Returns:
    A dataset_utils.Dataset object.
  """

  # Setup: General
  del rng

  # 1. Process information.
  p_idx = jax.process_index()  # current process index
  p_cnt = jax.process_count()  # process count (number of processes)
  # 2. dataset and data type information.
  dataset_configs = config.dataset_configs  # get dataset configurations.

  # Dataset names
  # TODO(girishvn): Make these non-hardcoded if this is used consistently.
  dataset_name_act = 'lsm_prod/lsm_300min_600_activities_balanced'
  dataset_name_mood = 'lsm_prod/lsm_300min_2000_mood_balanced'
  # This is only used to access the time-feature indicies.
  # These are the same for both activity and mood datasets.
  dataset_name = dataset_name_act

  dtype = getattr(tf, dtype_str)  # data dtype
  if eval_batch_size is None:  # set eval batch size
    eval_batch_size = batch_size

  # Setup: Mapping functions.
  # 1. Fewshot filter examples.
  samples_per_class = config.fewshot_samples_per_class
  fewshot_filter_fn = functools.partial(
      fewshot_filter_examples, n=samples_per_class,
  )

  # 2. Preprocessing, augmentation, and cropping/padding functions.
  preprocess_fn = functools.partial(
      lsm_mood_vs_activity_dataset.preprocess_example,
      dataset_name=dataset_name, dtype=dtype
  )
  # 2. Augmentation function.
  augment_fn = functools.partial(
      lsm_dataset_utils.augment_example,
      augmentations=config.get('train_augmentations', []),
  )
  # 3. Crop and pad features and time features to be patch size compatible.
  crop_and_pad_fn = functools.partial(
      lsm_dataset_utils.patch_compatible_resize_example,
      patch_size=config.model.patches.size
  )

  # Setup: Data splits.
  # 1. Train split: Get the entire or a subset of the training set.
  train_split_name = dataset_configs.get('train_split', 'train')
  num_train_samples = dataset_configs.get('train_num_samples', None)
  if num_train_samples:
    train_split = f'{train_split_name}[:{num_train_samples}]'
  else:
    train_split = train_split_name

  # 2. Validation split: Use the ENTIRE test split for validation.
  eval_split_name = dataset_configs.get('eval_split', 'test')
  val_split = eval_split_name

  # 3. Per-process split: Split splits evenly per worker).
  train_split_range = tfds.even_splits(split=train_split, n=p_cnt)[p_idx]
  val_split_range = tfds.even_splits(split=val_split, n=p_cnt)[p_idx]

  # 4. Load dataset splits.
  # Activity dataset
  train_act_ds = tfds.load(
      dataset_name_act,
      data_dir=data_dir,
      split=train_split_range,
      shuffle_files=False,  # NOTE: train shuffle is done below.
  )
  val_act_ds = tfds.load(
      dataset_name_act,
      data_dir=data_dir,
      split=val_split_range,
      shuffle_files=False,
  )
  # Mood dataset
  train_mood_ds = tfds.load(
      dataset_name_mood,
      data_dir=data_dir,
      split=train_split_range,
      shuffle_files=False,  # NOTE: train shuffle is done below.
  )
  val_mood_ds = tfds.load(
      dataset_name_mood,
      data_dir=data_dir,
      split=val_split_range,
      shuffle_files=False,
  )

  # Combine datasets.
  train_ds = train_act_ds.concatenate(train_mood_ds)
  val_ds = val_act_ds.concatenate(val_mood_ds)

  logging.info(  # pylint:disable=logging-fstring-interpolation
      f'Loaded train, val, splits {p_idx}/{p_cnt} from '
      f'{dataset_name_act} and {dataset_name_mood}.'
  )

  # Data processing and preperation.
  # 1. Enable multi threaded workers.
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  train_ds = train_ds.with_options(options)
  val_ds = val_ds.with_options(options)

  # 1a. Fewshot filter examples.
  train_ds = fewshot_filter_fn(ds=train_ds)

  # 2. Preprocessing: Applied before `ds.cache()` to re-use it.
  train_ds = train_ds.map(
      preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.map(
      preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )

  # 3. Cache datasets: This can signficantly speed up training.
  if dataset_configs.cache_dataset:
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

  # 4. Data preperation (repetition, shuffling, augmentations, batching, etc.).
  repeat_ds = dataset_configs.get('repeat_data', True)

  # 4a. Train: repeat, augment, crop/pad, shuffle, and batch.
  if repeat_ds:
    train_ds = train_ds.repeat()  # repeat
  # NOTE: Train augmentations are done after repeat for true randomness.
  if config.use_train_augmentations:
    train_ds = train_ds.map(  # train data augmentations
        augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
  train_ds = train_ds.map(  # crop/pad for perfect patching
      crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  shuffle_buffer_size = shuffle_buffer_size or (8 * batch_size)
  train_ds = train_ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)  # shuffle
  train_ds = train_ds.batch(batch_size, drop_remainder=True)  # batch
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # prefetch

  # 4b. Validation: crop/pad, batch, and repeat.
  val_ds = val_ds.map(  # crop/pad for perfect patching
      crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.batch(batch_size, drop_remainder=False)  # batch
  if repeat_ds:
    val_ds = val_ds.repeat()  # repeat
  val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

  # Other mappings
  # 0. Dataset service.
  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError(
          'Using dataset service with a random seed causes each '
          'worker to produce exactly the same data. Add '
          'config.shuffle_seed = None to your config if you '
          'want to run with dataset service.'
      )
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)
    logging.info('Using the tf.data service at %s', dataset_service_address)

  # 1. Set up batch padding: If batch remainders are NOT dropped batches may be
  # padded to allow for an enough patches to contain all samples.
  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=True,
      batch_size=batch_size,
      inputs_key='input_signal',
  )
  maybe_pad_batches_eval = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=False,
      batch_size=eval_batch_size,
      inputs_key='input_signal',
  )

  # 2. Set up batch sharding: Shard batches to be processed by multiple devices.
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)

  # 3. Apply other mappings and Iter dataset
  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(maybe_pad_batches_train, train_iter)
  train_iter = map(shard_batches, train_iter)

  val_iter = iter(val_ds)
  val_iter = map(dataset_utils.tf_to_numpy, val_iter)
  val_iter = map(maybe_pad_batches_eval, val_iter)
  val_iter = map(shard_batches, val_iter)

  # Save meta data
  info = tfds.builder(dataset_name, data_dir=data_dir, try_gcs=True).info
  input_shape = tuple([-1] + list(info.features['input_signal'].shape))
  train_num_act_samples = dataset_utils.get_num_examples(
      dataset=dataset_name_act, split=train_split, data_dir=data_dir
  )
  train_num_mood_samples = dataset_utils.get_num_examples(
      dataset=dataset_name_mood, split=train_split, data_dir=data_dir
  )
  eval_num_act_samples = dataset_utils.get_num_examples(
      dataset=dataset_name_act, split=val_split, data_dir=data_dir
  )
  eval_num_mood_samples = dataset_utils.get_num_examples(
      dataset=dataset_name_mood, split=val_split, data_dir=data_dir
  )
  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': int(2 * samples_per_class),
      'num_val_examples': eval_num_act_samples + eval_num_mood_samples,
      'num_test_examples': 0,
      'input_dtype': getattr(jnp, dtype_str),
      # The following two fields are set as defaults and may be updated in the
      # update_metadata function below.
      'target_is_onehot': True,
      'num_classes': 2,
      'label_names': ['mood', 'activity'],
      'label_counts': [train_num_mood_samples, train_num_act_samples]
  }
  # Update metadata to reflect preprocessing, and paddings
  # (Changes in shape, and features).
  meta_data.update(
      update_metadata(
          meta_data,
          dataset_name=dataset_name,
          patch_size=config.model.patches.size,
          samples_per_class=samples_per_class,
      )
  )

  # Return dataset structure.
  return dataset_utils.Dataset(train_iter, val_iter, None, meta_data)
