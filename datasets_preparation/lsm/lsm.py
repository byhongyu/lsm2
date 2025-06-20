"""LSM Tier v2 Dataset."""

import logging
import random
import time
from typing import Any

import apache_beam as beam
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from typing_extensions import override

from google3.medical.waveforms.modelling.lsm.datasets.lsm import constants
from google3.medical.waveforms.modelling.lsm.datasets.lsm import experiment_constants
from google3.medical.waveforms.modelling.lsm.datasets.lsm import sensors
from google3.pyglib import gfile


_DESCRIPTION = """LSM Tier 2 Dataset / V2"""

_CITATION = """LSM Tier 2 Dataset"""


def _init_feature_dict() -> dict[str, Any]:
  """Initializes the feature dict."""

  features_dict = {
      constants.TFExampleKey.INPUT.value: tf.string,
      constants.TFExampleKey.MASK.value: tf.string,
      constants.TFExampleKey.MISSINGNESS_RATIO.value: tf.float32,
      constants.TFExampleKey.USER_ID.value: tf.string,
      constants.TFExampleKey.KEY.value: tf.string,
      constants.TFExampleKey.CAPTION.value: tf.string,
  }

  return features_dict


def _serialize_numpy_array(array: np.ndarray) -> bytes:
  """Serializes a numpy array."""
  tensor = tf.convert_to_tensor(array)
  return tf.io.serialize_tensor(tensor).numpy()


def get_example_generator(user_id: str) -> Any:
  """Returns an example generator."""

  logging.info('Retrieving Data for user %s', user_id)
  try:
    timestamp_millis = int(time.time() * 1000)
    random.seed(timestamp_millis)
    for i, (_, data) in enumerate(
        sensors.window(
            [user_id],
            '%d%s' % (experiment_constants.WINDOW_SIZE, 'min'),
            's',
            experiment_constants.NUMBER_OF_SENSOR_FEATURES,
        )
    ):
      key = '%d_%d_%s' % (
          random.randint(0, 1000000000),
          i,
          user_id,
      )  # get unique key
      result = {
          constants.TFExampleKey.INPUT.value: _serialize_numpy_array(
              data[constants.TFExampleKey.INPUT.value]
          ),
          constants.TFExampleKey.MASK.value: _serialize_numpy_array(
              data[constants.TFExampleKey.MASK.value]
          ),
          constants.TFExampleKey.MISSINGNESS_RATIO.value: data[
              constants.TFExampleKey.MISSINGNESS_RATIO.value
          ],
          constants.TFExampleKey.USER_ID.value: data[
              constants.TFExampleKey.USER_ID.value
          ],
          constants.TFExampleKey.KEY.value: data[
              constants.TFExampleKey.KEY.value
          ],
          constants.TFExampleKey.CAPTION.value: data[
              constants.TFExampleKey.CAPTION.value
          ],
      }
      yield key, result
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        'Failed to retrieve data for user %s, error: %s', user_id, str(e)
    )


class Lsm(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for afib dataset."""

  VERSION = tfds.core.Version('2.0.0')
  RELEASE_NOTES = {
      '2.0.0': 'Initial release.',
  }

  @override
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(_init_feature_dict()),
        supervised_keys=None,
        citation=_CITATION,
    )

  def _split_generators(
      self, dl_manager: tfds.download.DownloadManager, pipeline: beam.Pipeline
  ) -> Any:
    """Returns SplitGenerators."""

    # read ids list from csv
    with gfile.Open(experiment_constants.TRAIN_ID_CSV, 'r') as f:
      train_ids = pd.read_csv(f)['session_id']
    with gfile.Open(experiment_constants.VALID_ID_CSV, 'r') as f:
      val_ids = pd.read_csv(f)['session_id']
    if experiment_constants.NUMBER_OF_SESSIONS != -1:
      train_ids = train_ids[: experiment_constants.NUMBER_OF_SESSIONS]
    if experiment_constants.VALID_ONLY:
      return {
          'valid': (
              pipeline
              | 'GenerateExamplesValid' >> self._generate_examples(val_ids)
          ),
      }
    elif experiment_constants.TRAIN_ONLY:
      return {
          'train': (
              pipeline
              | 'GenerateExamplesTrain' >> self._generate_examples(train_ids)
          ),
      }
    else:
      return {
          'train': (
              pipeline
              | 'GenerateExamplesTrain' >> self._generate_examples(train_ids)
          ),
          'valid': (
              pipeline
              | 'GenerateExamplesValid' >> self._generate_examples(val_ids)
          ),
      }

  def _generate_examples(self, ids: list[str]) -> Any:
    """Yields examples."""
    return beam.Create(ids) | 'GetExampleGenerator' >> beam.FlatMap(
        get_example_generator
    )
