"""Gets a dataset and formats it for distributed training and evaluation.

Adapted in part from a combination of the following files:
google3/third_party/py/scenic/dataset_lib/cifar10_dataset.py
google3/third_party/py/scenic/dataset_lib/dataset_utils.py
"""

from typing import Any, Optional

from absl import logging
import jax.numpy as jnp
import jax.profiler
import ml_collections  # pylint: disable=unused-import
from scenic.dataset_lib import dataset_utils

# Dataset Builders
# NOTE: LSM V1 dataset have been deprecated and moved to deprecated_datasets/.
# LSM V2 Datasets.
from google3.experimental.largesensormodels.scenic.datasets import lsm_v2_metabolic_health_dataset
from google3.experimental.largesensormodels.scenic.datasets import lsm_v2_pretraining_dataset


def get_dataset_builder(dataset_name: str):
  """Returns the dataset builder function for the given dataset name."""

  # 1. LSM V2 Metabolic Health Datasets.
  if dataset_name in [
      'metabolic_tfrecords_24h_missingness_80',
  ]:
    dataset_builder = (
        lsm_v2_metabolic_health_dataset.get_metabolic_health_dataset
    )

  # 2. LSM V2 Pretrain Datasets.
  elif 'lsm_v2' in dataset_name:
    dataset_builder = lsm_v2_pretraining_dataset.get_lsm_v2_dataset

  # UNSUPPORTED DATASETS.
  else:
    raise ValueError(
        f'Unsupported dataset: {dataset_name}. '
        'Check the dataset exists, is named correctly, and is not deprecated.'
    )

  return dataset_builder


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

  # Get dataset name and builder.
  dataset_path_name = config.dataset_configs.dataset

  if '/' in dataset_path_name:  # Used for V1 Datasets.
    dataset_name = dataset_path_name.split('/')[1]
  else:  # Used for V2 Datasets.
    dataset_name = dataset_path_name

  dataset_builder = get_dataset_builder(dataset_name)

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
  shuffle_buffer_size = config.dataset_configs.get('shuffle_buffer_size', None)
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
