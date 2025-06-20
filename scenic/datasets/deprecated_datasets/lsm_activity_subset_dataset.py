"""Electrodes dataset data preprocesser and loader.

This is specifically for the 9 class subset of the activities 600 dataset.

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


def preprocess_example(example, dataset_name, dtype=tf.float32):
  """Preprocesses the given example.

  Adapted from google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py

  Args:
    example: dict; Example that has an 'image' and a 'label'.
    dataset_name: str; Name of the dataset. This is used to extract the
      datetime features.
    dtype: Tensorflow data type; Data type of the image.

  Returns:
    A preprocessed example.

  NOTE: This assumes that the image is in the shape [H, W, C],
    where H is the Time axis, and W is the feature axis.
  """
  dataset_name = dataset_name.split('/')[-1]
  features = tf.cast(example['input_signal'], dtype=dtype)
  time_features = dataset_constants.lsm_dataset_constants[dataset_name].get(
      'datetime_features', None
  )

  if time_features is None:
    raise ValueError(dataset_name)

  # Split input into inputs and time-features
  feature_indices = list(range(features.shape[1]))
  if time_features is not None:

    # Get the inidices of datetime_features,
    # and split them from the indicies of other features.
    time_feature_indices = list(time_features['indices'])
    feature_indices = list(set(feature_indices) - set(time_features['indices']))
    time_feature_indices = tf.convert_to_tensor(time_feature_indices)
    feature_indices = tf.convert_to_tensor(feature_indices)

    # Using the above indices, split the feature tensor.
    time_features = tf.gather(features, time_feature_indices, axis=1)
    features = tf.gather(features, feature_indices, axis=1)
  else:
    time_features = None

  # Stress / Mood / Activity Labels:
  # A) Binary label of stress (0/1).
  stress_label = tf.cast(example['label'], dtype=tf.int32)
  # B) Boolean logs (True/False) of an logged exercise or mood event.
  # (exercise and mood events are mutally exclusive).
  exercise_log = example['metadata']['exercise_log']
  mood_log = example['metadata']['mood_log']
  # C) The log value (int 64 log code) for an excercise or mood event.
  # NOTE: that exercise and mood events DO NOT occur simultaneously
  log_value = tf.cast(example['metadata']['log_value'], tf.int32)

  # Return preprocessed feature and desired labels.
  # A) If activities or mood dataset: the log value is indexed [0, n classes],
  # one-hot encoded, and returned as the label.
  if ('activities' in dataset_name or 'mood' in dataset_name):
    # One hot encode the log value.
    # a) offset value of log_value - an artifact of dataset creation.
    log_value_offset = tf.cast(
        dataset_constants.lsm_dataset_constants[dataset_name][
            'label_value_offset'
        ],
        tf.int32
    )
    # b) list of possible labels (label_values) for a dataset.
    log_value_label_list = tf.convert_to_tensor(
        dataset_constants.lsm_dataset_constants[dataset_name]['label_values']
    )
    # c) offset log_value.
    log_value_label_list = log_value_label_list - log_value_offset
    n_classes = len(log_value_label_list)  # number of classes in label map
    # d) generate label map.
    lookup_initializer = tf.lookup.KeyValueTensorInitializer(
        keys=log_value_label_list, values=tf.range(n_classes)
    )
    label_map = tf.lookup.StaticHashTable(lookup_initializer, default_value=-1)
    # e) get label index from label map.
    label_idx = label_map.lookup(log_value)
    return {
        'input_signal': features,
        'datetime_signal': time_features,
        'label': tf.one_hot(label_idx, n_classes),
        'exercise_log': exercise_log,
        'mood_log': mood_log,
        'log_value': log_value,
    }

  # B) If stress dataset: the stress_label is one-hot encoded,
  # and returned as the label.
  elif 'stress' in dataset_name:
    stress_label = tf.cast(stress_label, tf.int32)
    return {
        'input_signal': features,
        'datetime_signal': time_features,
        'label': tf.one_hot(stress_label, 2)
    }

  # C) This is used for pretraining datasets.
  else:
    return {
        'input_signal': features,
        'datetime_signal': time_features,
        'stress_label': stress_label,
        'exercise_log': exercise_log,
        'mood_log': mood_log,
        'log_value': log_value,
    }


def augment_example(example, augmentations):
  """Applies augmentations (stretch, flip, noise) to the features."""

  augmented_feat = example['input_signal']
  height, width, _ = augmented_feat.shape

  # Stretch (along time/height axis).
  if 'stretch' in augmentations:
    apply_stretch = tf.random.uniform([], minval=0, maxval=1)
    if apply_stretch >= 0.5:
      stretch = tf.random.uniform([], minval=1.0, maxval=1.5)
      stretched_height = int(height * stretch)
      augmented_feat = tf.image.resize(
          augmented_feat, size=[int(stretched_height), int(width)]
      )
      offset_height = stretched_height - height
      augmented_feat = tf.image.crop_to_bounding_box(
          image=augmented_feat,
          target_height=height,
          target_width=width,
          offset_height=offset_height,
          offset_width=0,
      )

      # TODO(girishvn): apply translate?
      augmented_feat = augmented_feat[
          -1 * height :, :, :
      ]  # crop to original size

  # Flip (along time/height axis).
  if 'flip' in augmentations:
    apply_flip = tf.random.uniform([], minval=0, maxval=1)
    if apply_flip >= 0.5:
      augmented_feat = tf.image.flip_up_down(augmented_feat)

  # Noise (gaussian).
  if 'noise' in augmentations:
    apply_noise = tf.random.uniform([], minval=0, maxval=1)
    if apply_noise >= 0.5:
      noise_std = tf.random.uniform([], minval=0.0, maxval=0.5)
      noise = tf.random.normal(
          shape=tf.shape(augmented_feat), mean=0.0, stddev=noise_std
      )
      augmented_feat += noise

  example['input_signal'] = augmented_feat
  return example


def update_metadata(metadata, dataset_name, patch_size):
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

  return metadata_update


def filter_log_values(example, allowed_labels):
  label = example['metadata']['log_value']
  label = tf.cast(tf.reshape(label, []), tf.int32)
  keep_example = tf.reduce_any(tf.math.equal(label, allowed_labels))
  return keep_example


def get_electrodes_dataset(
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
    dataset_name=None,  # 'lsm_prod/lsm_300min_10M_impute'
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
  dataset_name = dataset_configs.get('dataset', dataset_name)  # get ds name
  dtype = getattr(tf, dtype_str)  # data dtype
  if eval_batch_size is None:  # set eval batch size
    eval_batch_size = batch_size

  if dataset_name != 'lsm_prod/lsm_300min_600_activities_9class_subset':
    raise ValueError(f'Unsupported dataset: {dataset_name}.')

  # MORE SET UP
  dataset_key = dataset_name.split('/')[-1]
  dataset_load_name = '/'.join(
      (
          'lsm_prod',
          dataset_constants.lsm_dataset_constants[dataset_key]['dataset_name']
      )
  )

  # Setup: Mapping functions.
  # 0. Filter labels
  allowed_labels = tf.convert_to_tensor(
      dataset_constants.lsm_dataset_constants[dataset_key]['label_values'],
      dtype=tf.int32
  )
  log_value_offset = tf.cast(
      dataset_constants.lsm_dataset_constants[dataset_key][
          'label_value_offset'
      ],
      tf.int32
  )
  allowed_labels = allowed_labels - log_value_offset
  filter_fn = functools.partial(
      filter_log_values, allowed_labels=allowed_labels
  )

  # 1. Preprocessing, augmentation, and cropping/padding functions.
  preprocess_fn = functools.partial(
      preprocess_example, dataset_name=dataset_name, dtype=dtype
  )
  # 2. Augmentation function.
  augment_fn = functools.partial(
      augment_example,
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

  # 2. Validation / Test splits: Split the test split into validation and
  # test sets. (50% - 50% split).
  eval_split_name = dataset_configs.get('eval_split', 'test')

  # 3. Per-process split: Split splits evenly per worker).
  train_split_range = tfds.even_splits(split=train_split, n=p_cnt)[p_idx]
  val_split_range = tfds.even_splits(split=eval_split_name, n=p_cnt)[p_idx]

  # 4. Load dataset splits.
  train_ds = tfds.load(
      dataset_load_name,
      data_dir=data_dir,
      split=train_split_range,
      shuffle_files=False,  # NOTE: train shuffle is done below.
  )
  val_ds = tfds.load(
      dataset_load_name,
      data_dir=data_dir,
      split=val_split_range,
      shuffle_files=False,
  )
  logging.info(  # pylint:disable=logging-fstring-interpolation
      f'Loaded train, val, and test split {p_idx}/{p_cnt} from {dataset_name}.'
  )

  # Data processing and preperation.
  # 1. Enable multi threaded workers.
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  train_ds = train_ds.with_options(options)
  val_ds = val_ds.with_options(options)

  # 2.a. Filter examples by log value.
  train_ds = train_ds.filter(filter_fn)
  val_ds = val_ds.filter(filter_fn)

  # 2.b. Preprocessing: Applied before `ds.cache()` to re-use it.
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
  info = tfds.builder(dataset_load_name, data_dir=data_dir, try_gcs=True).info
  input_shape = tuple([-1] + list(info.features['input_signal'].shape))
  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': 2470,
      'num_val_examples': 566,
      'num_test_examples': None,
      'input_dtype': getattr(jnp, dtype_str),
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
          dataset_name=dataset_name,
          patch_size=config.model.patches.size,
      )
  )

  # Return dataset structure.
  return dataset_utils.Dataset(train_iter, val_iter, None, meta_data)
