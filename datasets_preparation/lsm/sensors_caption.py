"""Creating Caption for Sensor-LM data."""

from typing import Dict, List
import numpy as np
from google3.medical.waveforms.modelling.lsm.datasets.lsm import sensor_constants as constants


# Grouped channels for caption generation.
channel_groups = {
    'Heart': {
        'members': [
            'HR',
            # 'hr_at_rest_mean',  # high corr
            # 'hrv_rr_80th_percentile_mean',  # high corr
            # 'hrv_rr_20th_percentile_mean',  # high corr
            'hrv_rr_median',
            'hrv_shannon_entropy_rr',
            # 'hrv_shannon_entropy_rrd',  # high corr
            # 'rmssd_percentile_0595',  # high corr
            'sdnn_percentile_0595',
        ],
        'names': [
            'heart rate',
            # 'hr at rest mean',
            # 'hrv rr 80th percentile',
            # 'hrv rr 20th percentile',
            'hrv rr',
            'hrv shannon entropy rr',
            # 'hrv shannon entropy rrd',
            # 'rmssd percentile mean',
            'sdnn percentile',
        ],
    },
    'Activity': {
        'members': [
            'steps',
            'jerk_auto',
            'log_energy',
            # 'covariance',
            # 'log_energy_ratio',
            # 'zero_crossing_std',
            # 'zero_crossing_avg',
            # 'axis_mean',
            # 'altim_std',
            'kurtosis',
        ],
        'names': [
            'steps',
            'jerk',
            'log energy',
            # 'covariance',
            # 'log energy ratio',
            # 'zero crossing std',
            # 'zero crossing avg',
            # 'axis mean',
            # 'altim std',
            'kurtosis',
        ],
    },
    'Sleep': {'members': ['sleep_coefficient'], 'names': ['sleep coefficient']},
    'EDA': {
        'members': [
            'eda_level_real',
            # 'leads_contact_counts',
            # 'ceda_slope_real_micro_siemens',
            'skin_temperature_slope',
            'wrist_temperatures',
        ],
        'names': [
            'eda level',
            # 'leads contact counts',
            # 'ceda slope real micro siemens',
            'skin temperature slope',
            'wrist temperatures',
        ],
    },
}


def denormalize_sensor_values(
    x: np.ndarray,
    labels: List[str],
    norm_params: Dict[str, list[float]],
) -> np.ndarray:
  """Denormalizes sensor values.

  Args:
      x: A np.ndarray of shape [len(features), N], where N is the time length,
        and features are the name of features of the channels.
      labels: A list of feature labels.
      norm_params: A dictionary of normalization parameters.

  Returns:
      The denormalized sensor values.
  """

  # denormalize the sensor values
  original_x = np.zeros_like(x)
  for i in range(len(labels)):
    original_x[:, i] = (
        x[:, i] * norm_params[labels[i]][1] + norm_params[labels[i]][0]
    )

  # make 'steps' and 'sleep_coefficient' values non-negative
  for i, label in enumerate(labels):
    if label in ['steps', 'sleep_coefficient']:
      original_x[:, i] = np.maximum(0, original_x[:, i])

  return original_x


def generate_caption(
    x_raw: np.ndarray, mask: np.ndarray, caption_with_impute_vals: bool = True
) -> str:
  """Generate a caption describing the mean (or sum) of each channel, grouped by category.

  Args:
      x_raw: A np.ndarray of shape [len(features), N], where N is the time
        length, and features are the name of features of the channels. Note that
        the input tensort representes the raw data, not the normalized data.
      mask: A np.ndarray of the same shape as x, where 1 indicates missingness.
      caption_with_impute_vals: If True, imputed values will be used in the
        caption.

  Returns:
      A formatted string describing each channel, grouped by category.
  """

  labels = constants.FEATURES_TO_INCLUDE
  norm_params = constants.NORMALIZATION_PARAMETERS

  if x_raw.shape[1] != len(labels):
    raise ValueError(
        f'Input tensor must have {len(labels)} channels, but got'
        f' {x_raw.shape[1]}.'
    )

  if mask is not None and not caption_with_impute_vals:
    if mask.shape != x_raw.shape:
      raise ValueError(
          f'Mask shape {mask.shape} must match input shape {x_raw.shape}.'
      )
    # set imputed values to NaN for exclusion in computations
    x_masked = np.where(mask == 1, np.nan, x_raw)
  else:
    x_masked = x_raw

  # denormalize the sensor values (ignore mask for now)
  original_x = denormalize_sensor_values(x_masked, labels, norm_params)

  # calculate mean, max, min, std on denormalized data
  channel_stats = []
  for i in range(len(labels)):
    channel_data = original_x[:, i]
    channel_stats.append({
        'mean': np.nanmean(channel_data),
        'max': np.nanmax(channel_data),
        'min': np.nanmin(channel_data),
        'std': np.nanstd(channel_data),
    })

  feature_stats_dict = dict(zip(labels, channel_stats))

  caption_parts = []
  for category, group in channel_groups.items():
    category_parts = []
    for name, member in zip(group['names'], group['members']):
      stats = feature_stats_dict.get(member, {})
      if stats:
        mean_val = stats.get('mean', np.nan)
        max_val = stats.get('max', np.nan)
        min_val = stats.get('min', np.nan)
        std_val = stats.get('std', np.nan)

        if all(
            not np.isnan(val) for val in [mean_val, max_val, min_val, std_val]
        ):
          category_parts.append(
              f'{name} mean, max, min, std are {mean_val:.1f}, {max_val:.1f},'
              f' {min_val:.1f}, {std_val:.1f}'
          )
    if category_parts:
      caption_parts.append(
          f'For {category}, ' + ', '.join(category_parts) + '.'
      )

  caption = ' '.join(caption_parts)
  return caption
