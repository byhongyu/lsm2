"""Electrodes dataset data preprocesser and loader.

Adapted from a combination of the following files:
google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py
google3/third_party/py/scenic/dataset_lib/dataset_utils.py
"""

import functools
from typing import Any, Optional

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

  return metadata_update


def get_combined_electrodes_pretrain_dataset(
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
    dataset_name=None,
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
  dataset_name = dataset_configs.get('dataset', dataset_name)  # get ds name
  dtype = getattr(tf, dtype_str)  # data dtype
  if eval_batch_size is None:  # set eval batch size
    eval_batch_size = batch_size

  # Setup: Mapping functions.
  # 1. Preprocessing, augmentation, and cropping/padding functions.
  preprocess_fn = functools.partial(
      lsm_v1_dataset.preprocess_example,
      dataset_name=dataset_name, dtype=dtype
  )
  # 2. Augmentation function.
  augment_fn = functools.partial(
      lsm_dataset_utils.augment_example,
      augmentations=config.get('train_augmentations', []),
      seed=tf_aug_rng,
  )
  # 3. Crop and pad features and time features to be patch size compatible.
  crop_and_pad_fn = functools.partial(
      lsm_dataset_utils.patch_compatible_resize_example,
      patch_size=config.model.patches.size
  )

  # Setup: Data splits.
  loaded_dataset1 = 'lsm_prod/lsm_300min_pretraining_165K_n10'
  loaded_dataset2 = 'lsm_prod/lsm_300min_10M_impute_deidentified'

  num_train_samples = dataset_configs.get('train_num_samples', None)
  train_split_name = 'train'
  val_split_name = 'test'

  # Total samples per data split.
  train1_total_samples = dataset_constants.lsm_dataset_constants[
      loaded_dataset1.split('/')[1]
  ]['num_train_examples']
  train2_total_samples = dataset_constants.lsm_dataset_constants[
      loaded_dataset2.split('/')[1]
  ]['num_train_examples']
  train3_total_samples = dataset_constants.lsm_dataset_constants[  # pylint:disable=unused-variable
      loaded_dataset2.split('/')[1]
  ]['num_test_examples']

  # 1. Train split: Get the data slices per dataset.
  if num_train_samples:
    # Only pull data from the 165k dataset.
    if num_train_samples <= train1_total_samples:
      train_split1 = f'{train_split_name}[:{num_train_samples}]'
      train_split2 = None
      train_split3 = None

    # Get the full 165K dataset, and some portion of the 6M pretrain set.
    else:
      train_split1 = train_split_name
      remaining_samples = num_train_samples - train1_total_samples

      if remaining_samples <= train2_total_samples:
        train_split2 = f'{train_split_name}[:{remaining_samples}]'
        train_split3 = None
      else:
        train3_samples = remaining_samples - train2_total_samples
        train_split2 = train_split_name
        train_split3 = f'{val_split_name}[:{train3_samples}]'


  # If num_train_samples not specified get the full dataset.
  else:
    train_split1 = train_split_name
    train_split2 = train_split_name
    train_split3 = val_split_name

  # 2. Validation split: get the entire test set.
  val_split = val_split_name

  # 3. Per-process split: Split splits evenly per worker).
  # Train DS 1.
  train_split1_range = tfds.even_splits(split=train_split1, n=p_cnt)[p_idx]
  # Train DS 2.
  if train_split2 is not None:
    train_split2_range = tfds.even_splits(split=train_split2, n=p_cnt)[p_idx]
  else:
    train_split2_range = None
    # Train DS 3.
  if train_split3 is not None:
    train_split3_range = tfds.even_splits(split=train_split3, n=p_cnt)[p_idx]
  else:
    train_split3_range = None

  # Validation.
  val_split_range = tfds.even_splits(split=val_split, n=p_cnt)[p_idx]

  # 4. Load dataset splits.
  # Load training splits
  # Train split from lsm_300min_pretraining_165K_n10
  train_ds1 = tfds.load(
      loaded_dataset1,
      data_dir=data_dir,
      split=train_split1_range,
      shuffle_files=False,  # NOTE: train shuffle is done below.
  )
  # Train split from lsm_300min_10M_impute
  if train_split2_range is not None:
    train_ds2 = tfds.load(
        loaded_dataset2,
        data_dir=data_dir,
        split=train_split2_range,
        shuffle_files=False,  # NOTE: train shuffle is done below.
    )
  else:
    train_ds2 = None
  if train_split3_range is not None:
    train_ds3 = tfds.load(
        loaded_dataset2,
        data_dir=data_dir,
        split=train_split3_range,
        shuffle_files=False,  # NOTE: train shuffle is done below.
    )
  else:
    train_ds3 = None

  # Combine train splits into a single dataset.
  train_ds = train_ds1
  if train_ds2 is not None:
    train_ds = train_ds.concatenate(train_ds2)
  if train_ds3 is not None:
    train_ds = train_ds.concatenate(train_ds3)

  # Load eval splits.
  val_ds = tfds.load(
      loaded_dataset1,
      data_dir=data_dir,
      split=val_split_range,
      shuffle_files=False,
  )
  logging.info(  # pylint:disable=logging-fstring-interpolation
      f'Loaded train, and val split {p_idx}/{p_cnt} from {dataset_name}.'
  )

  # Data processing and preperation.
  # 0. Enable multi threaded workers.
  options = tf.data.Options()
  options.threading.private_threadpool_size = 48
  train_ds = train_ds.with_options(options)
  val_ds = val_ds.with_options(options)

  # 1. Preprocessing: Applied before `ds.cache()` to re-use it.
  train_ds = train_ds.map(
      preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.map(
      preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )

  # 2. Cache datasets: This can signficantly speed up training.
  if dataset_configs.cache_dataset:
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

  # 3. Data preperation (repetition, shuffling, augmentations, batching, etc.).
  repeat_ds = dataset_configs.get('repeat_data', True)

  # 3a. Train: repeat, augment, crop/pad, shuffle, and batch.
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

  # 3b. Validation: crop/pad, batch, and repeat.
  val_ds = val_ds.map(  # crop/pad for perfect patching
      crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.batch(batch_size, drop_remainder=False)  # batch
  if repeat_ds:
    val_ds = val_ds.repeat()  # repeat
  val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

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
  info = tfds.builder(loaded_dataset1, data_dir=data_dir, try_gcs=True).info
  input_shape = tuple([-1] + list(info.features['input_signal'].shape))

  num_train_examples_ds1 = dataset_utils.get_num_examples(
      dataset=loaded_dataset1, split=train_split1, data_dir=data_dir
  )
  num_eval_examples_ds1 = dataset_utils.get_num_examples(
      dataset=loaded_dataset1, split=val_split, data_dir=data_dir
  )

  if train_split2 is not None:
    num_train_examples_ds2 = dataset_utils.get_num_examples(
        dataset=loaded_dataset2, split=train_split2, data_dir=data_dir
    )
  else:
    num_train_examples_ds2 = 0

  if train_split3 is not None:
    num_eval_examples_ds2 = dataset_utils.get_num_examples(
        dataset=loaded_dataset2, split=val_split, data_dir=data_dir
    )
  else:
    num_eval_examples_ds2 = 0

  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': (
          num_train_examples_ds1 +
          num_train_examples_ds2 +
          num_eval_examples_ds2
      ),
      'num_val_examples': num_eval_examples_ds1,
      'num_test_examples': 0,
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


def get_dataset(
    config: Any,
    data_rng: jnp.ndarray,
    *,
    num_local_shards: Optional[int] = None,
    dataset_service_address: Optional[str] = None,
    **kwargs: Any,
) -> dataset_utils.Dataset:
  """Adapted from: google3/third_party/py/scenic/train_lib/train_utils.py."""

  # Get device count
  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  # Set the dataset builder functions
  # Get list of supported, non-deprecated datasets.
  dataset_suported_list = ['lsm_300min_pretraining_8M_combined']
  dataset_name = config.dataset_configs.dataset
  if dataset_name.split('/')[1] in dataset_suported_list:
    dataset_builder = get_combined_electrodes_pretrain_dataset
  else:
    raise ValueError(f'Dataset {dataset_name} is not supported.')

  # Get batch size
  batch_size = config.batch_size
  if batch_size % device_count > 0:
    raise ValueError(
        f'Batch size ({batch_size}) must be divisible by the '
        f'number of devices ({device_count})'
    )

  local_batch_size = batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  # Get shuffle seed - ensure it exists
  shuffle_seed = config.get('shuffle_seed', None)
  if dataset_service_address and shuffle_seed is not None:
    raise ValueError(
        'Using dataset service with a random seed causes each '
        'worker to produce exactly the same data. Add '
        'config.shuffle_seed = None to your config if you want '
        'to run with dataset service.'
    )

  # Get shuffle buffer size.
  shuffle_buffer_size = config.dataset_configs.shuffle_buffer_size
  # Local shard count.
  num_local_shards = num_local_shards or jax.local_device_count()

  # Build the dataset
  ds = dataset_builder(
      config=config,
      num_shards=num_local_shards,
      batch_size=local_batch_size,
      dtype_str=config.data_dtype_str,
      shuffle_seed=shuffle_seed,
      rng=data_rng,
      shuffle_buffer_size=shuffle_buffer_size,
      dataset_service_address=dataset_service_address,
      **kwargs,
  )

  return ds

