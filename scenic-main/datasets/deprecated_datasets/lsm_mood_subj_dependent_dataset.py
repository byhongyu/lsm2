"""Electrodes dataset data preprocesser and loader.

Adapted from a combination of the following files:
google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py
google3/third_party/py/scenic/dataset_lib/dataset_utils.py
"""

import collections
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
from google3.experimental.largesensormodels.scenic.datasets import lsm_tiny_dataset as lsm_v1_dataset


def filter_allowed_subjects(example, allowed_subjects):
  """Filter out examples where the label is not in allowed_labels."""
  subj = example['metadata']['ID']
  keep_example = tf.reduce_any(tf.math.equal(subj, allowed_subjects))
  return keep_example


def update_metadata(
    metadata, dataset_name, patch_size, dataset_configs
):
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

  # Update if dataset it one-hot-encoded or not.
  if 'activities' in dataset_name or 'mood' in dataset_name:
    metadata_update['target_is_onehot'] = True
    metadata_update['num_classes'] = len(
        dataset_constants.lsm_dataset_constants[dataset_name]['label_values']
    )
  elif 'stress' in dataset_name:
    metadata_update['target_is_onehot'] = True
    metadata_update['num_classes'] = 2

  # 4. Add dataset log values and log value names and number of classes.
  log_values = dataset_constants.lsm_dataset_constants[dataset_name].get(
      'label_values', None
  )
  log_value_names = dataset_constants.lsm_dataset_constants[dataset_name].get(
      'label_names', None
  )
  metadata_update['label_values'] = log_values
  metadata_update['label_names'] = log_value_names

  # 7. Update time cropping:
  start, end = dataset_configs.get('relative_time_window', (None, None))
  if end is None:
    end = 1
  if start is None:
    start = 0

  # Time Crop image based on horizon.
  # Get number of patches along time axis (h).
  p_h = patch_size[0]
  h = feat_shape_new[0]
  n_h = h // p_h
  start_idx = int(start * n_h) * p_h
  end_idx = int(end * n_h) * p_h
  metadata_update['input_shape'] = tuple(
      [-1] + [end_idx - start_idx] + list(feat_shape_new)[1:]
  )

  return metadata_update


def get_subject_dependent_mood_dataset(
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
    data_dir='/namespace/fitbit-medical-sandboxes/partner/encrypted/chr-ards-electrodes/deid/exp/dmcduff/ttl=6w/msa_1_5/lsm_tfds_datasets',
):
  """Gets and formats the Subject Dependent Mood Dataset.

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
    data_dir: str; Directory of the dataset.

  Returns:
    A dataset_utils.Dataset object.
  """

  # Setup: General
  if rng is None:
    rng = jax.random.PRNGKey(config.rng_seed)

  # 1. Process information.
  p_idx = jax.process_index()  # current process index
  p_cnt = jax.process_count()  # process count (number of processes)

  aug_rngs = jax.random.split(rng, p_cnt)  # per-device augmentation seeds
  aug_rng = aug_rngs[p_idx]  # device augmentation seed
  tf_aug_rng = aug_rng[0]  # jax random seeds are arrays, tf expects an int.
  del rng

  # 2. dataset and data type information.
  dataset_configs = config.dataset_configs  # get dataset configurations.
  dtype = getattr(tf, dtype_str)  # data dtype
  if eval_batch_size is None:  # set eval batch size
    eval_batch_size = batch_size

  # 3. Used dataset name.
  used_dataset_name = 'lsm_prod/lsm_300min_2000_mood_balanced'

  # 4. Repeat dataset.
  repeat_ds = dataset_configs.get('repeat_data', True)

  # Setup: Mapping functions.
  # 2. Preprocessing, augmentation, and cropping/padding functions.
  preprocess_fn = functools.partial(
      lsm_v1_dataset.preprocess_example,
      dataset_name=used_dataset_name,
      dtype=dtype
  )
  # 3. Augmentation function.
  augment_fn = functools.partial(
      lsm_dataset_utils.augment_example,
      augmentations=config.get('train_augmentations', []),
      seed=tf_aug_rng,
  )
  # 4. Crop and pad features and time features to be patch size compatible.
  crop_and_pad_fn = functools.partial(
      lsm_dataset_utils.patch_compatible_resize_example,
      patch_size=config.model.patches.size
  )

  # 5. Time crop data input
  start, end = dataset_configs.get('relative_time_window', (None, None))
  if (start is not None) or (end is not None):
    time_crop_examples = True
  else:
    time_crop_examples = False
  time_crop_fn = functools.partial(
      lsm_dataset_utils.time_crop_example,
      patch_size=config.model.patches.size,
      start=start,
      end=end
  )

  # Setup: Data splits.
  # Load dataset splits.
  train_ds = tfds.load(
      used_dataset_name,
      data_dir=data_dir,
      split='train',
      shuffle_files=False,  # NOTE: train shuffle is done below.
  )
  val_ds = tfds.load(
      used_dataset_name,
      data_dir=data_dir,
      split='test',
      shuffle_files=False,
  )
  logging.info(  # pylint:disable=logging-fstring-interpolation
      'Loaded combined train + val split '
      f'{p_idx}/{p_cnt} from {used_dataset_name}.'
  )
  ds = train_ds.concatenate(val_ds)

  # Data processing and preperation.
  # 0. Enable multi threaded workers.
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  # 1. Per-process split: Split splits evenly per worker).
  # Count samples per subject.
  subj_label_counts = collections.Counter()
  for d in ds:
    subj = d['metadata']['ID']
    subj_label_counts[subj.numpy().decode('utf-8')] += 1

  # Filter down to subjects with at least N samples.
  allowed_subjs = [
      subj for subj, count in subj_label_counts.items()
      if count >= dataset_configs.min_samples_per_subject
  ]
  filter_fn = functools.partial(
      filter_allowed_subjects, allowed_subjects=allowed_subjs
  )
  ds = ds.filter(filter_fn)

  # Split the data into train and val splits.
  # Splits each participants data evenly between train and val.
  def filter_by_subj(subj):
    return lambda x: tf.equal(
        x['metadata']['ID'], subj
    )

  # TODO(girishvn): should I shuffle the data before splitting?

  train_subj_splits, valid_subj_splits = [], []
  num_train_samples, num_val_samples = 0, 0
  for subj in allowed_subjs:
    subj_ds = ds.filter(filter_by_subj(subj))
    size_subj_ds = sum(1 for _ in subj_ds)

    train_size = int(0.8 * size_subj_ds)
    num_train_samples += train_size
    num_val_samples += size_subj_ds - train_size

    subj_train_split = subj_ds.take(train_size)
    subj_val_split = subj_ds.skip(train_size)

    train_subj_splits.append(subj_train_split)
    valid_subj_splits.append(subj_val_split)

  # Concat class datasets.
  train_ds = train_subj_splits[0]
  val_ds = valid_subj_splits[0]
  for i in range(1, len(allowed_subjs)):
    train_ds = train_ds.concatenate(train_subj_splits[i])
    val_ds = val_ds.concatenate(valid_subj_splits[i])

  # Get samples per class
  spc = collections.Counter()
  for d in train_ds:
    log_val = int(d['metadata']['log_value'])
    spc[log_val] += 1

  spc_labels = tf.convert_to_tensor(list(spc.keys()))
  spc_label_counts = tf.convert_to_tensor(list(spc.values()))

  # Get mood log values
  dataset_key = used_dataset_name.split('/')[-1]
  offset = tf.cast(
      dataset_constants.lsm_dataset_constants[dataset_key][
          'label_value_offset'
      ],
      tf.int32
  )
  log_val_list = tf.convert_to_tensor(
      dataset_constants.lsm_dataset_constants[dataset_key]['label_values']
  )
  log_val_list = log_val_list - offset  # offset value

  sorted_indices_tensor2 = tf.argsort(spc_labels)
  matching_indices = tf.argsort(tf.argsort(log_val_list))
  mapping = tf.gather(sorted_indices_tensor2, matching_indices)
  label_counts = tf.gather(spc_label_counts, mapping)

  # Split dataset over host devices.
  train_ds = train_ds.shard(p_cnt, p_idx)
  val_ds = val_ds.shard(p_cnt, p_idx)

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

  # 4 Train repeats and augmentations.
  if repeat_ds:
    train_ds = train_ds.repeat()  # repeat
  # NOTE: Train augmentations are done after repeat for true randomness.
  if config.use_train_augmentations:
    train_ds = train_ds.map(  # train data augmentations
        augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  # 5. Crop and pad for perfect patching.
  train_ds = train_ds.map(  # crop/pad for perfect patching
      crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.map(  # crop/pad for perfect patching
      crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )

  # 6. Time crop input data.
  if time_crop_examples:
    train_ds = train_ds.map(
        time_crop_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.map(
        time_crop_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  # 7. Data preperation (shuffling, augmentations, batching, eval repeat, etc.).
  # 7a. Train: Shuffle, batch, prefetch
  shuffle_buffer_size = shuffle_buffer_size or (8 * batch_size)
  train_ds = train_ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)  # shuffle
  train_ds = train_ds.batch(batch_size, drop_remainder=True)  # batch
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # prefetch

  # 7b. Validation: Batch, Repeat, Prefetch
  val_ds = val_ds.batch(batch_size, drop_remainder=False)  # batch
  if repeat_ds:
    val_ds = val_ds.repeat()  # repeat
  val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)  # prefetch

  # Ensure that no seed is set if dataset_service_address is defined.
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

  # Other mappings
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
  info = tfds.builder(used_dataset_name, data_dir=data_dir, try_gcs=True).info
  input_shape = tuple([-1] + list(info.features['input_signal'].shape))
  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': num_train_samples,
      'num_val_examples': num_val_samples,
      'num_test_examples': 0,
      'input_dtype': getattr(jnp, dtype_str),
      'label_counts': label_counts,
      # The following two fields are set as defaults and may be updated in the
      # update_metadata function below.
      'target_is_onehot': False,
      'num_classes': None,
  }

  # Update metadata to reflect preprocessing, and paddings
  # (Changes in shape, and features).
  meta_data.update(
      update_metadata(
          meta_data,
          dataset_name=used_dataset_name,
          patch_size=config.model.patches.size,
          dataset_configs=dataset_configs,
      )
  )

  # Return dataset structure.
  return dataset_utils.Dataset(train_iter, val_iter, None, meta_data)
