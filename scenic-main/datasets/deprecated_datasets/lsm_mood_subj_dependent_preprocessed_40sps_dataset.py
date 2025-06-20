"""Electrodes dataset data preprocesser and loader.

Adapted from a combination of the following files:
google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py
google3/third_party/py/scenic/dataset_lib/dataset_utils.py

NOTE: This dataset is a HACKY implementation of the
lsm_mood_subj_dependent_dataset which specifically loads a preprpocessed dataset
where subject dependent splits are already created, and where only subjects with
40+ samples are included.

This was created as part of the ICLR '25 Rebuttal for the LSM paper.
This hacky implementation is necessary as lsm_mood_subj_dependent_dataset.py is
made extremely slow on XM (causing idle failures) due to the need to traverse
the dataset to create the subject dependent splits.
The dataset is created in the following colab:
experimental/largesensormodels/notebooks/dataset_exploration/lsm_downstream_task_dataset_explorer.ipynb

If you are interested in using this dataset, please consider using the
lsm_mood_subj_dependent_dataset.py instead, and / or re-implemting this.
"""


import functools
import os
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

  Adapted from /largesensormodels/scenic/datasets/dataset_utils.py.
  This function is modified to work with the pre-processed dataset, which has
  slight differences from the original dataset (e.g. not have 'metadata' field).

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
  stress_label = tf.cast(example['label'][0], dtype=tf.int32)  # pylint: disable=unused-variable
  # B) Boolean logs (True/False) of an logged exercise or mood event.
  # (exercise and mood events are mutally exclusive).
  exercise_log = example['exercise_log'][0]
  mood_log = example['mood_log'][0]
  # C) The log value (int 64 log code) for an excercise or mood event.
  # NOTE: that exercise and mood events DO NOT occur simultaneously
  log_value = tf.cast(example['log_value'][0], tf.int32)

  # Return preprocessed feature and desired labels.
  # A) If activities or mood dataset: the log value is indexed [0, n classes],
  # one-hot encoded, and returned as the label.
  # a) offset value of log_value - an artifact of dataset creation.
  log_value_offset = tf.cast(
      dataset_constants.lsm_dataset_constants[dataset_name][
          'label_value_offset'
      ],
      tf.int32
  )
  # b) list of possible labels (log_values) for a dataset.
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


def parse_tfexample_fn(example):
  """Parses features from serialized tf example."""
  # The dataset has more labels than we use.
  feature_spec = {
      'input_signal': tf.io.FixedLenFeature(
          shape=[300, 30, 1], dtype=tf.float32
      ),
      'label': tf.io.FixedLenFeature(
          shape=1, dtype=tf.int64
      ),
      'exercise_log': tf.io.FixedLenFeature(
          shape=1, dtype=tf.int64
      ),
      'mood_log': tf.io.FixedLenFeature(
          shape=1, dtype=tf.int64
      ),
      'log_value': tf.io.FixedLenFeature(
          shape=1, dtype=tf.int64
      ),
  }
  parsed_example = tf.io.parse_single_example(example, feature_spec)
  parsed_example['exercise_log'] = tf.cast(
      parsed_example['exercise_log'], tf.bool
  )
  parsed_example['mood_log'] = tf.cast(parsed_example['mood_log'], tf.bool)
  return parsed_example


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
  metadata_update['target_is_onehot'] = True
  metadata_update['num_classes'] = len(
      dataset_constants.lsm_dataset_constants[dataset_name]['label_values']
  )

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

  # START HARDCODED SECTION
  # As explained in the file header: this is hardcoded to read from a
  # pre-processed version of the subject dependent mood dataset.

  processed_data_dir = (
      '/namespace/fitbit-medical-sandboxes/partner/encrypted/'
      'chr-ards-electrodes/deid/exp/girishvn/ttl=6w/lsm_processed_datasets/'
  )
  train_fname = 'processed_subj_dependent_mood_train.tfrecord'
  test_fname = 'processed_subj_dependent_mood_test.tfrecord'

  # Reference dataset name, used to query for dataset_constants.
  used_dataset_name = 'lsm_prod/lsm_300min_2000_mood_balanced'

  # Pre-computed label counts per class.
  label_counts = [786, 460, 408, 767, 1132]

  # Pre-computed train and test sample counts.
  num_train_samples = 3553
  num_val_samples = 1154
  # END HARDCODED SECTION

  train_fpath = os.path.join(processed_data_dir, train_fname)
  val_fpath = os.path.join(processed_data_dir, test_fname)

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

  # 4. Repeat dataset.
  repeat_ds = dataset_configs.get('repeat_data', True)

  # Setup: Mapping functions.
  # 2. Preprocessing, augmentation, and cropping/padding functions.
  preprocess_fn = functools.partial(
      preprocess_example,
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
  train_ds = tf.data.TFRecordDataset(train_fpath)
  val_ds = tf.data.TFRecordDataset(val_fpath)
  train_ds = train_ds.map(parse_tfexample_fn)
  val_ds = val_ds.map(parse_tfexample_fn)

  # Data processing and preperation.
  # 0. Enable multi threaded workers.
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  train_ds = train_ds.with_options(options)
  val_ds = val_ds.with_options(options)

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
