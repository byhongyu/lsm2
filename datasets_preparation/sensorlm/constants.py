"""Constants for SensorLM datasets."""

import enum


class TFExampleKey(enum.Enum):
  """Enum for TFExample keys."""

  INPUT = 'input_signal'
  MASK = 'mask'
  MISSINGNESS_RATIO = 'missingness_ratio'
  USER_ID = 'user_id'
  KEY = 'key'
  CAPTION = 'caption'
