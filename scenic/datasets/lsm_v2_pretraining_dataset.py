"""LSM V2 Pretraining Multi-Trainer Dataset.

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


def patch_compatible_resize_example(
    example: tf.Tensor,
    patch_size: tuple[int, int],
    feature_shape: tuple[int, int, int],
):
  """Crops and pads features to allow for a integer number of patches.

  NOTE: This assumes that the image is in the shape [H, W, C], where H is the
  Time axis which can be cropped, and W is the feature axis which can be padded.

  NOTE: This should be applied AFTER augmentations as to ensure that noise is
    not applied to zero-padding.

  Args:
    example: A dictionary of inputs containing at least the 'input_signal',
      'labels', and possibly 'datetime_signal' fields.
    patch_size: tuple; Size of the patches to extract from the image (H, W).
    feature_shape: tuple; Shape of the feature in order [Time, Sensors, 1]

  Returns:
    A dictionary of inputs containing at least the 'input_signal', 'labels',
      and possibly 'datetime_signal' fields. Where 'input_signal', and possibly
      'datetime_signal' fields are H cropped and W padded.
  """
  # Parse inputs
  features = example['input_signal']
  imputation_mask = example['imputation_mask']  # binary mask for NaNs

  # Crop time axis (h) and pad feature axis (w)
  crop_h, pad_w, _ = lsm_dataset_utils.get_height_crop_width_pad(
      feature_shape, patch_size
  )
  features = features[crop_h[0] :, :, :]
  features = tf.pad(
      features,
      paddings=[[0, 0], pad_w, [0, 0]],
      mode='CONSTANT',
      constant_values=0,
  )
  # Apply the same crop and pad to the imputation mask
  # NOTE: This is done ONLY if the imputation mask is not None.
  # There are some datasets (e.g. discriminative datasets) where the imputation
  # mask is does not exist.
  if example['imputation_mask'] is not None:
    imputation_mask = imputation_mask[crop_h[0] :, :, :]
    imputation_mask = tf.pad(
        imputation_mask,
        paddings=[[0, 0], pad_w, [0, 0]],
        mode='CONSTANT',
        constant_values=0,
    )

  # Update the example
  example['input_signal'] = features
  example['datetime_signal'] = None
  example['imputation_mask'] = imputation_mask
  return example


def preprocess_example(example, dataset_name, config, dtype=tf.float32):  # pylint: disable=unused-argument
  """Preprocesses the given example.

  Adapted from google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py

  Args:
    example: dict; Example that has an 'image' and a 'label'.
    dataset_name: str; Name of the dataset. This is used to extract the datetime
      features.
    config: ml_collections.ConfigDict; Config for the experiment.
    dtype: Tensorflow data type; Data type of the image.

  Returns:
    A preprocessed example.

  NOTE: This assumes that the image is in the shape [H, W, C],
    where H is the Time axis, and W is the feature axis.
  """
  features = tf.io.parse_tensor(example['input_signal'], out_type=tf.double)
  features = tf.cast(features, dtype)
  features = tf.expand_dims(features, axis=-1)
  imputation_mask = tf.io.parse_tensor(example['mask'], out_type=tf.bool)
  imputation_mask = tf.expand_dims(imputation_mask, axis=-1)
  imputation_ratio = example['missingness_ratio']
  imputation_ratio = tf.squeeze(imputation_ratio)

  # If dumping data retain the string keys
  if config.get('enable_dump_mode', False):
    return {
        'input_signal': features,
        'datetime_signal': None,
        'label': None,
        'exercise_log': None,
        'mood_log': None,
        'log_value': None,
        'imputation_mask': imputation_mask,
        'imputation_ratio': imputation_ratio,
        'user_id': example['user_id'],
        'key': example['key'],
    }
  else:
    return {
        'input_signal': features,
        'datetime_signal': None,
        'label': None,
        'exercise_log': None,
        'mood_log': None,
        'log_value': None,
        'imputation_mask': imputation_mask,
        'imputation_ratio': imputation_ratio,
    }


def update_metadata(
    metadata, dataset_name, patch_size, filter_examples, dataset_configs  # pylint: disable=unused-argument
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
    metadata_update['datime_valid_feats'] = tuple(
        valid_time_feat_mask
    )  # @TODO(@girishvn): update this

  else:
    metadata_update['datetime_input_shape'] = None
    metadata_update['datime_valid_feats'] = None

  # 6. Update time cropping:
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


def get_lsm_v2_dataset(
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
    data_dir='/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/lsm_v2/datasets/tfds',
):
  """Gets and formats the LSM V2 Pre-Training Dataset.

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
  train_dataset_name = dataset_configs.dataset  # get ds name
  train_data_dir = data_dir + '/' + train_dataset_name
  if dataset_configs.get('valid_dataset', None) is None:
    valid_dataset_name = dataset_configs.dataset
    valid_data_dir = data_dir + '/' + valid_dataset_name
  # Necessary for the specialized directory of the "concatenated" dataset.
  # TODO(xliucs, xumax) clean up later
  else:
    valid_dataset_name = dataset_configs.valid_dataset  # get ds name
    valid_data_dir = data_dir + '_test' + '/' + valid_dataset_name
  dtype = getattr(tf, dtype_str)  # data dtype
  print(f'train_dataset_name: {train_dataset_name}')
  print(f'valid_dataset_name: {valid_dataset_name}')
  print(f'train_data_dir: {train_data_dir}')
  print(f'valid_data_dir: {valid_data_dir}')
  if eval_batch_size is None:  # set eval batch size
    eval_batch_size = batch_size

  # 4. Repeat dataset.
  repeat_ds = dataset_configs.get('repeat_data', True)

  # Setup: Mapping functions.
  # 1. Preprocessing, augmentation, and cropping/padding functions.
  train_preprocess_fn = functools.partial(
      preprocess_example,
      dataset_name=train_dataset_name,
      config=config,
      dtype=dtype,
  )
  valid_preprocess_fn = functools.partial(
      preprocess_example,
      dataset_name=valid_dataset_name,
      config=config,
      dtype=dtype,
  )

  # 2. Augmentation function.
  # augment_fn = functools.partial(
  #     lsm_dataset_utils.augment_example,
  #     augmentations=config.get('train_augmentations', []),
  #     seed=tf_aug_rng,
  # )

  # 3. Crop and pad features and time features to be patch size compatible.
  crop_and_pad_fn = functools.partial(
      patch_compatible_resize_example,
      patch_size=config.model.patcher_config.patchsize,
      feature_shape=dataset_constants.lsm_dataset_constants[train_dataset_name][
          'feature_shape'
      ],
  )

  # 4. Time crop data input
  # start, end = dataset_configs.get('relative_time_window', (None, None))
  # if (start is not None) or (end is not None):
  #   time_crop_examples = True
  # else:
  #   time_crop_examples = False
  # time_crop_fn = functools.partial(
  #     time_crop_example,
  #     patch_size=config.model.patcher_config.patchsize,
  #     start=start,
  #     end=end,
  # )

  # Setup: Data splits.
  # 1. Train split: Get the entire or a subset of the training set.
  train_split_name = dataset_configs.get('train_split', 'train')
  test_split_name = dataset_configs.get('eval_split', 'valid')
  num_train_samples = dataset_configs.get('train_num_samples', None)
  num_test_samples = dataset_configs.get('eval_num_samples', None)
  if num_train_samples:
    train_split = f'{train_split_name}[:{num_train_samples}]'
  elif num_train_samples == -1:
    train_split = train_split_name
  else:
    train_split = train_split_name

  if num_test_samples:
    val_split = f'{test_split_name}[:{num_test_samples}]'
  else:
    val_split = test_split_name

  # NOTE: Validation and test splits were previously evenly split from the same
  # data split, as per the below code. We have opted to only use a validation
  # split as all experiments use a static number of training steps.
  # eval_split_name = dataset_configs.get('eval_split', 'test')
  # val_split, test_split = tfds.even_splits(split=eval_split_name, n=2)

  # 3. Per-process split: Split splits evenly per worker).
  train_split_range = tfds.even_splits(split=train_split, n=p_cnt)[p_idx]
  val_split_range = tfds.even_splits(split=val_split, n=p_cnt)[p_idx]

  # 4. Load dataset splits.
  train_ds = tfds.load(
      'lsm',
      data_dir=train_data_dir,
      split=train_split_range,
      shuffle_files=False,  # NOTE: train shuffle is done below.
  )
  logging.info(  # pylint:disable=logging-fstring-interpolation
      f'Loaded train {p_idx}/{p_cnt} from {train_dataset_name}.'
  )

  try:  # new balanced dataset for evals
    valdata_type = 'LsmMissingBalanced'
    val_ds = tfds.load(
        valdata_type,
        data_dir=valid_data_dir,
        split=val_split_range,
        shuffle_files=False,
    )
  except tfds.core.registered.DatasetNotFoundError:  # old dataset for evals
    valdata_type = 'lsm'
    val_ds = tfds.load(
        valdata_type,
        data_dir=data_dir,
        split=val_split_range,
        shuffle_files=False,
    )

  logging.info(  # pylint:disable=logging-fstring-interpolation
      f'Loaded valid split {p_idx}/{p_cnt} from {valid_dataset_name}.'
  )
  # Data processing and preperation.
  # 0. Enable multi threaded workers.
  options = tf.data.Options()
  options.threading.private_threadpool_size = 0
  train_ds = train_ds.with_options(options)
  val_ds = val_ds.with_options(options)

  # 1. Preprocessing: Applied before `ds.cache()` to re-use it.
  train_ds = train_ds.map(
      train_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.map(
      valid_preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  # 2. Cache datasets: This can signficantly speed up training.
  if dataset_configs.cache_dataset:
    if num_train_samples <= 10000:
      train_ds = train_ds.cache()
    val_ds = val_ds.cache()

  # 3 Train repeats and augmentations.
  if repeat_ds:
    train_ds = train_ds.repeat()  # repeat
  # NOTE: Train augmentations are done after repeat for true randomness.
  # if config.use_train_augmentations:
  #   train_ds = train_ds.map(  # train data augmentations
  #       augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  #   )

  # 4. Crop and pad for perfect patching.
  train_ds = train_ds.map(  # crop/pad for perfect patching
      crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  val_ds = val_ds.map(  # crop/pad for perfect patching
      crop_and_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )

  # 4.5 Construct Masks
  # Note that not all pretrain methods (e.g. non-masking) have a masker_config.
  if (
      config.get('masker_config', False) and
      config.masker_config.on_cpu
  ):
    mask_fn = functools.partial(
        lsm_dataset_utils.mask_example,
        input_size=dataset_constants.lsm_dataset_constants[train_dataset_name][
            'feature_shape'
        ],
        patch_size=config.model.patcher_config.patchsize,
        masker_config=config.masker_config,
        seed=tf_aug_rng,
    )

    train_ds = train_ds.map(
        mask_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    val_ds = val_ds.map(
        mask_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  # 5. Time crop input data.
  # if time_crop_examples:
  #   train_ds = train_ds.map(
  #       time_crop_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  #   )
  #   val_ds = val_ds.map(
  #       time_crop_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
  #   )

  # 6. Data preperation (shuffling, augmentations, batching, eval repeat, etc.).
  # 6a. Train: Shuffle, batch, prefetch
  shuffle_buffer_size = shuffle_buffer_size or (8 * batch_size)
  train_ds = train_ds.shuffle(shuffle_buffer_size, seed=shuffle_seed)  # shuffle
  train_ds = train_ds.batch(batch_size, drop_remainder=True)  # batch
  train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # prefetch

  # 6b. Validation: Batch, Repeat, Prefetch
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
  input_shape = dataset_constants.lsm_dataset_constants[train_dataset_name][
      'feature_shape'
  ]
  input_shape = tuple([-1] + list(input_shape))
  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': dataset_utils.get_num_examples(
          dataset='lsm', split=train_split, data_dir=train_data_dir
      ),
      'num_val_examples': dataset_utils.get_num_examples(
          dataset=valdata_type, split=val_split, data_dir=valid_data_dir
      ),
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
          dataset_name=train_dataset_name,
          patch_size=config.model.patcher_config.patchsize,
          filter_examples=False,
          dataset_configs=dataset_configs,
      )
  )

  # Return dataset structure.
  return dataset_utils.Dataset(train_iter, val_iter, None, meta_data)
