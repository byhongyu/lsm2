"""Metabolic Health dataset.

Adapted from a combination of the following files:
google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py
google3/third_party/py/scenic/dataset_lib/dataset_utils.py
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

from google3.experimental.largesensormodels.scenic.datasets import dataset_constants
from google3.experimental.largesensormodels.scenic.datasets import dataset_utils as lsm_dataset_utils  # pylint: disable=unused-import
from google3.experimental.largesensormodels.scenic.datasets import lsm_v2_pretraining_dataset

from google3.pyglib import gfile


def task_preprocess_example(example, task_name, softmax_loss=True):
  """Preprocesses the given example for a given downstream task."""

  unprocessed_label = example[task_name]

  # Preprocess binary tasks.
  # 'binray' to account for misspelled label 'homa_ir_binray'.
  # 'respiratory' is a binary task though not named as such.
  if (
      'binary' in task_name or
      'binray' in task_name or
      task_name == 'respiratory'
  ):
    # If using a softmax loss, treat binary task as 2-class multi-class task.
    if softmax_loss:
      label = tf.one_hot(unprocessed_label, 2)
    # Treat as a 1-class binary detection task.
    else:
      label = unprocessed_label

  else:
    raise ValueError(f'Task name {task_name} is not yet supported.')

  return {
      'input_signal': example['input_signal'],
      'input_metadata': example['input_metadata'],
      'label': label,
      'unprocessed_label': unprocessed_label,
      'imputation_mask': example['imputation_mask'],
      'imputation_ratio': example['imputation_ratio'],
      'subject_id': example['subject_id'],
  }


def preprocess_example(example, dataset_name, dtype=tf.float32):  # pylint: disable=unused-argument
  """Preprocesses the given example.

  Adapted from google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py

  Args:
    example: dict; Example that has an 'input_signal' key as well as keys
      associated with the labels. In this function the 'datetime_signal',
      'label', 'imputation_mask', and 'imputation_ratio' keys are added.
      Additional examples keys can be found in the parse_metabolic_tfrecord
      function.
    dataset_name: str; Name of the dataset. This is used to extract the datetime
      features.
    dtype: Tensorflow data type; Data type of the image.

  Returns:
    A preprocessed example (dictionary) with added 'datetime_signal', 'label',
    'imputation_mask', and 'imputation_ratio' keys.

  NOTE: This assumes that the input signal is of the shape [T, M] and adds a new
    axis C, where T is the Time axis, and M is the feature axis, and C is the
    channel axis (C=1).
  """

  # Type cast and add channel dimension to input signal.
  feature = tf.cast(example['input_signal'], dtype)
  feature = tf.expand_dims(feature, axis=-1)  # Add channel dimension.
  encoded_meta_data = tf.cast(example['input_metadata'], dtype)
  imputation_mask = tf.expand_dims(example['imputation_mask'], axis=-1)
  example['input_signal'] = feature
  example['input_metadata'] = encoded_meta_data
  example['datetime_signal'] = None
  example['imputation_mask'] = imputation_mask
  example['imputation_ratio'] = None
  return example


def parse_metabolic_tfrecord(example):
  """Parse example from feature.

  This record parsing is needed as most other datasets are saved as TFDS
  datasets and thus do not need a feature schema to read in the dataset.
  This metablic health dataset requires a schema to be parsed from the
  tfrecords.

  Args:
    example: tf.Example; Example to parse.

  Returns:
    A parsed example.
  """

  # FEATURE WITH ALL LABELS
  feature = {
      # Float Labels
      'bmi': tf.io.FixedLenFeature([], tf.float32),  # body mass index
      'homa_ir': tf.io.FixedLenFeature([], tf.float32),  # insulin resistance
      'apri': tf.io.FixedLenFeature([], tf.float32),  # liver health score
      'msss': tf.io.FixedLenFeature([], tf.float32),

      # Integer Labels
      # TODO(dmcduff): spelling error in generated dataset
      'homa_ir_binray': tf.io.FixedLenFeature([], tf.int64),
      'msss_binary': tf.io.FixedLenFeature([], tf.int64),
      'hypertension_binary': tf.io.FixedLenFeature([], tf.int64),
      'hyperlipidemia_binary': tf.io.FixedLenFeature([], tf.int64),
      'cardiovascular_binary': tf.io.FixedLenFeature([], tf.int64),
      'diabetes_binary': tf.io.FixedLenFeature([], tf.int64),
      'anxiety_binary': tf.io.FixedLenFeature([], tf.int64),
      'age': tf.io.FixedLenFeature([], tf.int64),
      'gender': tf.io.FixedLenFeature([], tf.int64),

      # Int Labels
      'respiratory': tf.io.FixedLenFeature([], tf.int64),
      'kidney_disease': tf.io.FixedLenFeature([], tf.int64),

      # String Labels
      'regular_menstruation_str': tf.io.RaggedFeature(tf.string),
      'smoker_str': tf.io.RaggedFeature(tf.string),
      'diabetes_type_str': tf.io.RaggedFeature(tf.string),
      'alcohol_str': tf.io.RaggedFeature(tf.string),
      'medications_str': tf.io.RaggedFeature(tf.string),

      # Input Signal
      'array_raw': tf.io.FixedLenFeature([], tf.string),

      # Mask of Input Signal
      'array_mask': tf.io.FixedLenFeature([], tf.string),
  }

  fpath = example['filepath']
  example = example['example']

  # Parse the example
  example = tf.io.parse_single_example(example, feature)

  # Decode the input signal and mask
  example['input_signal'] = tf.io.parse_tensor(
      example['array_raw'], out_type=tf.double
  )
  example.pop('array_raw')  # remove raw input signal

  example['imputation_mask'] = tf.io.parse_tensor(
      example['array_mask'], out_type=tf.bool
  )
  example.pop('array_mask')  # remove raw input signal

  # Add a vector of metadata
  age = tf.cast(example['age'], tf.float32)
  gender = tf.cast(example['gender'], tf.float32)
  bmi = example['bmi']
  input_metadata = tf.stack([age, gender, bmi], axis=-1)
  example['input_metadata'] = input_metadata

  # get subject id from filename
  fpath_parts = tf.strings.split(fpath, '/')
  fname = fpath_parts[-1]
  fname = tf.strings.regex_replace(fname, '.tfrecords', '')  # remove .tfrecords
  fname_parts = tf.strings.split(fname, '_')
  subj_id_str = fname_parts[-1]  # get subject id from filename
  subj_id = tf.strings.to_number(subj_id_str, out_type=tf.int32)
  example['subject_id'] = subj_id

  return example


# TODO(girishvn): Move this to a common file, and limit its functionarlity to
# LSMv1 datasets.
def update_input_shapes(
    original_input_shape: tuple[int, int, int, int],
    patch_size: Optional[tuple[int, int]],
    relative_time_window: Optional[tuple[float, float]],
    dataset_name: str
):
  """Updates the input shape to reflect padding and cropping.

  Given the original input shape, this function updates the input shape to
  reflect padding and cropping. This function is used for both sensor features
  as well as datetime features.

  Args:
    original_input_shape: tuple[int, int, int, int]; The original input shape.
      Dimensions are [batch, time, features, 1].
    patch_size: Optional[tuple[int, int]]; The patch size.
    relative_time_window: Optional[tuple[int, int]]; The relative time window to
      (as a percentage of the total time) to crop the time dimension by.
    dataset_name: str; The name of the dataset.

  Returns:
    A dictionary containing the updated input shape and valid features for both
      sensor features and datetime features.
  """

  # 0. Set up: Get dataset constants dictionary.
  dataset_consts_dict = dataset_constants.lsm_dataset_constants[dataset_name]

  # 1. Get list of indices associated with datetime features in the input array.
  time_features = dataset_consts_dict.get('datetime_features', None)
  feature_shape = list(original_input_shape[1:])  # input shape sans batch dim
  feature_indices = list(range(feature_shape[1]))

  # 2. Split features from time series features
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

  # 3. Padding: Update shape to reflect padding (for perfect patching).
  # valid_feats arrays denote which features are valid (1) vs padded (0).
  # 3a. Update for sensor features
  if patch_size is not None:
    _, pad_w, feat_shape_new = lsm_dataset_utils.get_height_crop_width_pad(
        tuple(feature_shape), patch_size
    )
    valid_feat_mask = [0] * pad_w[0] + [1] * feature_shape[1] + [0] * pad_w[1]
    input_shape = tuple([-1] + list(feat_shape_new))
    input_valid_feats = tuple(valid_feat_mask)
  else:
    input_shape = tuple([-1] + list(feature_shape))
    input_valid_feats = tuple([1] * feature_shape[1])  # all features are valid

  # 3b. Update for datetime features
  # Time features exist.
  if time_features is not None:
    # Pad time features to be perfect patches.
    if patch_size is not None:
      _, time_pad_w, time_feature_shape_new = (
          lsm_dataset_utils.get_height_crop_width_pad(
              tuple(time_feature_shape), patch_size
          )
      )
      valid_time_feat_mask = (
          [0] * time_pad_w[0] +
          [1] * time_feature_shape[1] +
          [0] * time_pad_w[1]
      )
      datetime_input_shape = tuple([-1] + list(time_feature_shape_new))
      datime_valid_feats = tuple(valid_time_feat_mask)

    # No padding needed.
    else:
      datetime_input_shape = tuple([-1] + list(time_feature_shape))
      datime_valid_feats = tuple([1] * time_feature_shape[1])

  # No time features.
  else:
    datetime_input_shape = None
    datime_valid_feats = None

  # 4. Time Crop sensor matrix.
  # This crops the sensor matrix to a specific window along the time axis.
  # This is useful to isolated specific time-windows on the-fly for quicker
  # experiment iterations.
  if relative_time_window is not None:
    start, end = relative_time_window

    # a. If input is patched.
    if patch_size is not None:
      p_time = patch_size[0]  # Get number of patches along time axis (h).
      time = input_shape[1]  # input shape of Batch x Time x Features, 1
      n_time = time // p_time  # patches along time axis
      start_idx = int(start * n_time) * p_time
      end_idx = int(end * n_time) * p_time
      # new input shape is batch x cropped time x features x 1
      input_shape = tuple(
          [-1] + [end_idx - start_idx] + list(input_shape)[2:]
      )
    else:
      # b. If input is not patched.
      # Get start and end indices based on time cropping.
      time = input_shape[1]
      start_idx = int(start * time)
      end_idx = int(end * time)
      # new input shape is batch x cropped time x features x 1
      input_shape = tuple([-1] + [end_idx - start_idx] + list(input_shape)[2:])

  return {
      'input_shape': input_shape,
      'input_valid_feats': input_valid_feats,
      'datetime_input_shape': datetime_input_shape,
      'datetime_valid_feats': datime_valid_feats,
  }


def update_metadata(
    metadata, dataset_name, config  # pylint: disable=unused-argument
):
  """Update metadata to reflect resizing and addition of datetime features."""
  # 0. Setup: Get dataset name, feature shape, and possible datetime features.
  metadata_update = dict()
  dataset_name = dataset_name.split('/')[-1]
  dataset_consts_dict = dataset_constants.lsm_dataset_constants[dataset_name]
  dataset_configs = config.dataset_configs  # get dataset configurations.

  # 1. Update input shapes for features and datetime features.
  original_input_shape = metadata['input_shape']
  relative_time_window = dataset_configs.get('relative_time_window', None)
  patch_size = config.model.patcher_config.patchsize
  metadata_update.update(
      update_input_shapes(
          original_input_shape=original_input_shape,
          patch_size=patch_size,
          relative_time_window=relative_time_window,
          dataset_name=dataset_name
      )
  )

  # 1. Update task information / labels.
  task_name = dataset_configs.task_name
  metadata_update['task_name'] = task_name

  if task_name is not None:
    task_constants = dataset_consts_dict[task_name]

    # If the task is binary ('binary' in the task name)
    # 'binray' to account for misspelled label 'homa_ir_binray'.
    # 'respiratory' is a binary task though not named as such.
    if (
        'binary' in task_name or
        'binray' in task_name or
        task_name == 'respiratory'
    ):
      loss_name = config.classification_loss.loss_name
      # If using a softmax loss for binary classification two outputs needed.
      if 'softmax' in loss_name:
        metadata_update['target_is_onehot'] = True
        metadata_update['num_classes'] = 2
      # If a sigmoid loss is used, only one output needed.
      else:
        metadata_update['target_is_onehot'] = True
        metadata_update['num_classes'] = 1

    else:
      raise ValueError(f'Unsupported task name: {task_name}')

    # 3. Add dataset log values and log value names and number of classes.
    metadata_update['label_values'] = task_constants['label_values']
    metadata_update['label_names'] = task_constants['label_names']
    # 4. Add label weights.
    metadata_update['label_weights'] = task_constants['label_weights']
    # 5. Add label counts.
    metadata_update['label_counts'] = task_constants['label_counts']

  else:
    raise ValueError('Task name (dataset_configs.task_name) is not set.')

  return metadata_update


def get_metabolic_health_dataset(
    *,
    config: ml_collections.ConfigDict,
    num_shards: int,
    batch_size: int,
    eval_batch_size: Optional[int] = None,
    dtype_str: str = 'float32',
    shuffle_seed: int = 0,
    rng: Optional[jnp.ndarray] = None,
    shuffle_buffer_size: Optional[int] = None,
    dataset_service_address: Optional[str] = None,
    data_dir: Optional[str] = None,
):
  """Gets and formats the LSM Metabolic Health Downstream Dataset.

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
    A dataset_utils.Dataset object. The object contains train_iter, val_iter,
    test_iter, and meta_data. Each dataset iterator (e.g. train_iter, val_iter,
    test_iter) contains at minimum the 'input_signal' array, and optionally
    contains a number of label keys. These label keys are enumerated in
    parse_metabolic_tfrecord and preprocess_example.
  """
  # Setup: General
  del rng  # rng is currently unused (no random augmentations used).

  # 1. Process information.
  p_idx = jax.process_index()  # current process index
  p_cnt = jax.process_count()  # process count (number of processes)

  # 2. dataset and data type information.
  dataset_configs = config.dataset_configs  # get dataset configurations.
  dataset_name = dataset_configs.dataset  # get ds name
  if data_dir is None:  # Get data directory from dataset constants.
    data_dir = dataset_constants.lsm_dataset_constants[
        dataset_name
    ]['dataset_dir']
  # Repeat dataset flag. Not repeating the datasets (repeat_flag=False) is
  # helpful for examining and debugging the dataset.
  repeat_flag = dataset_configs.get('repeat_data', True)  # repeat dataset
  dtype = getattr(tf, dtype_str)  # data dtype
  if eval_batch_size is None:  # set eval batch size
    eval_batch_size = batch_size

  # Get dataset constants.
  dataset_consts_dict = dataset_constants.lsm_dataset_constants[dataset_name]

  # Setup: Data splits.
  if dataset_name in ['metabolic_tfrecords_24h_missingness_80']:
    # 1. Train / Val Split
    load_data_dir = os.path.join(data_dir, dataset_name)
    data_fpattern = os.path.join(load_data_dir, '*.tfrecords')
    filenames = gfile.Glob(data_fpattern)  # get dataset files
    filenames.sort()  # sort file names for deterministic split

    # Split into train and eval
    # Splitting is done based on subject id.
    ds_split_id = dataset_consts_dict['data_split_id']
    train_files = []
    val_files = []
    for f in filenames:
      _, fname = os.path.split(f)
      subj_id = int(fname.replace('.tfrecords', '').split('_')[-1])
      if subj_id < ds_split_id:
        train_files.append(f)
      else:
        val_files.append(f)

  else:
    raise ValueError(
        f'Dataset {dataset_name} is not supported by '
        'get_metabolic_health_dataset in lsm_v2_metabolic_health_dataset.py'
    )

  # Assert files exist and log file counts.
  assert train_files
  assert val_files
  logging.info('Number of train files: %d', len(train_files))
  logging.info('Number of val files: %d', len(val_files))

  # Convert to tf.data.Dataset
  train_files = tf.data.Dataset.from_tensor_slices(train_files)
  val_files = tf.data.Dataset.from_tensor_slices(val_files)

  # Interleave data examples, and add filepath information.
  train_ds = train_files.interleave(  # Get dataset
      lambda filename: tf.data.TFRecordDataset(filename).map(
          lambda example: {'filepath': filename, 'example': example}
      ),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  val_ds = val_files.interleave(
      lambda filename: tf.data.TFRecordDataset(filename).map(
          lambda example: {'filepath': filename, 'example': example}
      ),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  # Split dataset accross devices.
  # TODO(girishvn): should this be switched with .shard() instead?
  num_train_examples = dataset_consts_dict['num_train_examples']
  num_val_examples = dataset_consts_dict['num_test_examples']
  p_train_cnt = num_train_examples // p_cnt
  p_val_cnt = num_val_examples // p_cnt

  train_start = p_idx * p_train_cnt
  val_start = p_idx * p_val_cnt
  if p_idx == p_cnt - 1:
    train_ds = train_ds.skip(train_start)
    val_ds = val_ds.skip(val_start)
  else:
    train_ds = train_ds.skip(train_start).take(p_train_cnt)
    val_ds = val_ds.skip(val_start).take(p_val_cnt)

  # Data processing and preparation.
  # 0. Enable multi threaded workers.
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  train_ds = train_ds.with_options(options)
  val_ds = val_ds.with_options(options)

  # 0. Parse tfrecord.
  parse_fn = functools.partial(parse_metabolic_tfrecord)
  train_ds = train_ds.map(
      parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.map(
      parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  # 1. Preprocessing, augmentation, and cropping/padding functions.
  # Preprocessing: Applied before `ds.cache()` to re-use it.
  preprocess_fn = functools.partial(
      preprocess_example, dataset_name=dataset_name, dtype=dtype
  )
  train_ds = train_ds.map(
      preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.map(
      preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )

  # 2. Add task labels.
  task_name = dataset_configs.task_name
  loss_name = config.classification_loss.loss_name
  if task_name is not None:
    task_preprocess_fn = functools.partial(
        task_preprocess_example,
        task_name=task_name,
        softmax_loss=('softmax' in loss_name)
    )
    train_ds = train_ds.map(
        task_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.map(
        task_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  # 2. Cache datasets: This can signficantly speed up training.
  if dataset_configs.cache_dataset:
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

  # 3 Train repeats and augmentations.
  if repeat_flag:
    train_ds = train_ds.repeat()  # repeat

  # 3. Crop and pad features and time features to be patch size compatible.
  patch_size = config.model.patcher_config.patchsize
  if patch_size is not None:
    crop_and_pad_fn = functools.partial(
        lsm_v2_pretraining_dataset.patch_compatible_resize_example,
        patch_size=config.model.patcher_config.patchsize,
        feature_shape=dataset_consts_dict['feature_shape'],
    )
    train_ds = train_ds.map(  # crop/pad for perfect patching
        crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.map(  # crop/pad for perfect patching
        crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  # 4. Construct Token Masks on CPU.
  # TODO(xumax) code currently will silently work if imputation_mask is none
  # If CPU-bound masking is enabled and config.masker_config.inherited is
  # enabled, iterate through mask strategies and affirm that:
  if (
      config.get('masker_config') is not None and
      config.masker_config.on_cpu and
      config.masker_config.inherited
  ):
    # a. mask_probability is 0.
    for mask_strat in config.masker_config.maskstrategy_list:
      assert mask_strat.mask_probability == 0.0
    # b. strictmaskperc is 0.
    assert config.masker_config.strictmaskperc == 0.0

    # Take a single sample to get the input shape.
    input_size = train_ds.take(1).get_single_element()['input_signal'].shape
    mask_fn = functools.partial(
        lsm_dataset_utils.mask_example,
        input_size=input_size,
        patch_size=config.model.patcher_config.patchsize,
        masker_config=config.masker_config,
        seed=shuffle_seed,  # should not be using seed anyways so its fine
    )

    train_ds = train_ds.map(
        mask_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.map(
        mask_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  # 6. Data preperation (shuffling, augmentations, batching, eval repeat, etc.).
  # 6a. Train: Shuffle, batch, prefetch
  shuffle_buffer_size = shuffle_buffer_size or (8 * batch_size)
  train_ds = train_ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)
  train_ds = train_ds.batch(batch_size, drop_remainder=True)  # batch
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # prefetch

  # 6b. Validation: Batch, Repeat, Prefetch
  val_ds = val_ds.batch(batch_size, drop_remainder=False)  # batch
  if repeat_flag:
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
  original_input_shape = dataset_consts_dict['feature_shape']
  original_input_shape = tuple([-1] + list(original_input_shape))
  meta_data = {
      'input_shape': original_input_shape,
      'num_train_examples': num_train_examples,
      'num_val_examples': num_val_examples,
      'num_test_examples': 0,
      'input_dtype': getattr(jnp, dtype_str),
      # The following two fields are set as defaults and may be updated in the
      # update_metadata function below.
      'target_is_onehot': False,
      'num_classes': None,
  }
  # Update metadata to reflect preprocessing, and paddings
  # (Changes in shape, and features).
  # TODO(girishvn): add filtered examples functionality.
  meta_data.update(
      update_metadata(
          meta_data,
          dataset_name=dataset_name,
          config=config,
      )
  )

  # Return dataset structure.
  return dataset_utils.Dataset(train_iter, val_iter, None, meta_data)
