"""Utils for loading and processing LSM data."""

# TODO(dmcduff): please fix all pylint and pytype errors.
# pylint: disable=missing-class-docstring
# pylint: disable=no-self-argument
# pylint: disable=self-cls-assignment
# pylint: disable=g-doc-args
# pylint: disable=g-bare-generic
# pylint: disable=unused-argument
# pylint: disable=missing-function-docstring
# pylint: disable=unnecessary-lambda
# pylint: disable=unused-variable
# pylint: disable=g-importing-member

import abc
import dataclasses
import datetime
import functools as ft
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from google3.fitbit.research.sensing.common.colab import metadata_database_helpers
from google3.fitbit.research.sensing.common.infra.transforms import data_loading
from google3.fitbit.research.sensing.common.infra.utils import common_data_helper
from google3.fitbit.research.sensing.common.infra.utils import constants
from google3.fitbit.research.sensing.common.infra.utils import data_intermediates
from google3.fitbit.research.sensing.common.proto import data_key_pb2
from google3.fitbit.research.sensing.kereru.utils import data_loader
from google3.medical.waveforms.modelling.lsm.datasets.lsm import constants as lsm_constants
from google3.medical.waveforms.modelling.lsm.datasets.lsm import experiment_constants
from google3.medical.waveforms.modelling.lsm.datasets.lsm import sensor_constants
from google3.medical.waveforms.modelling.lsm.datasets.lsm import sensors_caption


def is_nan_percent_within_range(
    data: pd.DataFrame,
    percent_threshold_low: float,
    percent_threshold_high: float,
) -> bool:
  """Check if the total number of NaN values is within a range of thresholds."""
  total_values = data.size
  total_nan = data.isna().sum().sum()
  return (
      percent_threshold_high
      > (total_nan / total_values)
      > percent_threshold_low
  )


def check_if_break(
    dict_count: Dict[Tuple[float, float], int], threshold: int
) -> bool:
  """Check if the missingness ratio is within a range of thresholds."""
  for value in dict_count.values():
    if value < threshold:
      return False
  return True


def check_if_skip(
    missing_ratio: float,
    dict_count: Dict[Tuple[float, float], int],
    max_per_user_window_size: int,
) -> bool:
  """Check if the missingness ratio is within a range of thresholds."""
  for threshold_high, threshold_low in dict_count:
    if threshold_high > missing_ratio >= threshold_low:
      if (
          dict_count[(threshold_high, threshold_low)]
          == max_per_user_window_size
      ):
        return True
  return False


def _is_nan_percent_above(df: pd.DataFrame, percent_threshold: float) -> bool:
  """Check if the total number of NaN values is more than a threshold."""
  total_values = df.size
  total_nan = df.isna().sum().sum()
  return (total_nan / total_values) > percent_threshold


def _generate_missing_mask_ratio(
    data: pd.DataFrame,
) -> Tuple[np.ndarray, float]:
  """Generate a missing mask and missingness ratio for a numpy array."""
  nan_mask = data.isna()
  nan_mask = nan_mask.to_numpy()
  missingness_ratio = np.sum(nan_mask) / (nan_mask.shape[0] * nan_mask.shape[1])
  return nan_mask, missingness_ratio


def process_each_window(
    data: pd.DataFrame, missingness_ratio_threshold: float
) -> Tuple[str, np.ndarray, float]:
  """Process each sensor window."""
  if _is_nan_percent_above(data, missingness_ratio_threshold):
    return ('CONTINUE', None, None)
  elif data.empty:
    return ('CONTINUE', None, None)
  else:
    nan_mask, missingness_ratio = _generate_missing_mask_ratio(data)
    return ('PROCESS', nan_mask, missingness_ratio)


def process_each_window_with_stratified_users(
    data: pd.DataFrame,
    sensor_window_count_dict: Dict[Tuple[float, float], int],
    max_per_user_window_size: int,
    missingness_ratio_threshold: float,
) -> Tuple[str, np.ndarray, float]:
  """Process each sensor window with double threshold."""
  if data.empty:
    return ('CONTINUE', None, None)
  if check_if_break(
      sensor_window_count_dict,
      max_per_user_window_size,
  ):
    return ('BREAK', None, None)
  if _is_nan_percent_above(data, missingness_ratio_threshold):
    return ('CONTINUE', None, None)
  nan_mask, missingness_ratio = _generate_missing_mask_ratio(data)
  # Check if user already has window for the specified missingness range.
  if check_if_skip(
      missingness_ratio,
      sensor_window_count_dict,
      max_per_user_window_size,
  ):
    return ('CONTINUE', None, None)
  # Update the count of windows for the specified missingness range.
  for (
      threshold_high,
      threshold_low,
  ) in sensor_window_count_dict:
    if threshold_high > missingness_ratio >= threshold_low:
      sensor_window_count_dict[(threshold_high, threshold_low)] += 1
  return ('PROCESS', nan_mask, missingness_ratio)


def process_each_window_with_double_threshold(
    data: pd.DataFrame, missingness_ratio_threshold: float
) -> Tuple[str, np.ndarray, float]:
  """Process each sensor window with double threshold."""
  if data.empty:
    return ('CONTINUE', None, None)
  nan_mask, missingness_ratio = _generate_missing_mask_ratio(data)
  if (
      missingness_ratio_threshold
      > missingness_ratio
      > missingness_ratio_threshold - 0.1
  ):
    return ('PROCESS', nan_mask, missingness_ratio)
  else:
    return ('CONTINUE', None, None)


def filter_ceda(df: pd.DataFrame) -> pd.DataFrame:
  """Filter out CEDA values."""
  df.loc[df['eda_level_real'] > 60, 'eda_level_real'] = 60
  df.loc[df['eda_level_real'] < 0, 'eda_level_real'] = 0
  df.loc[df['eda_slope_real'] > 5, 'eda_slope_real'] = 5
  df.loc[df['eda_slope_real'] < -5, 'eda_slope_real'] = -5
  return df


def filter_wrist_temperatures(df: pd.DataFrame) -> pd.DataFrame:
  """Filter out wrist temperatures."""
  df['wrist_temperatures'] = df['wrist_temperatures'] / 20000
  df.loc[df['wrist_temperatures'] > 41, 'wrist_temperatures'] = 41
  df.loc[df['wrist_temperatures'] < 0, 'wrist_temperatures'] = np.nan
  return df


def filter_msa(df: pd.DataFrame) -> pd.DataFrame:
  """Filter out MSA values."""
  df.loc[df['hrv_percent_good'] < 20, 'hrv_rr_80th_percentile_mean'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'hrv_rr_20th_percentile_mean'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'hrv_rr_median'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'hrv_rr_mean'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'hrv_shannon_entropy_rr'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'hrv_shannon_entropy_rrd'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'hrv_percentage_of_nn_30'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'rmssd_percentile_0595'] = np.nan
  df.loc[df['hrv_percent_good'] < 20, 'sdnn_percentile_0595'] = np.nan
  df['hrv_shannon_entropy_rr'] = df['hrv_shannon_entropy_rr'] / 100
  df['hrv_shannon_entropy_rrd'] = df['hrv_shannon_entropy_rrd'] / 100
  df['hrv_percentage_of_nn_30'] = df['hrv_percentage_of_nn_30'] / 100
  df.loc[df['rmssd_percentile_0595'] > 125, 'rmssd_percentile_0595'] = 125
  df.loc[df['sdnn_percentile_0595'] > 125, 'sdnn_percentile_0595'] = 125
  df.loc[df['msa_probability'] == 102, 'msa_probability'] = np.nan
  df.loc[df['hr_at_rest_mean'] == 0, 'hr_at_rest_mean'] = np.nan
  df['hrv_percent_good'] = df['hrv_percent_good'] / 100
  df.loc[
      df['ceda_magnitude_real_micro_siemens'] > 60,
      'ceda_magnitude_real_micro_siemens',
  ] = np.nan
  df.loc[
      df['ceda_magnitude_real_micro_siemens'] < 0,
      'ceda_magnitude_real_micro_siemens',
  ] = 0
  df.loc[
      df['ceda_slope_real_micro_siemens'] > 5,
      'ceda_slope_real_micro_siemens',
  ] = 5
  df.loc[
      df['ceda_slope_real_micro_siemens'] < -5,
      'ceda_slope_real_micro_siemens',
  ] = -5
  df.loc[
      df['skin_temperature_magnitude'] == 1800,
      'skin_temperature_slope',
  ] = np.nan
  df.loc[
      df['ceda_magnitude_real_micro_siemens'] == 0,
      'ceda_slope_real_micro_siemens',
  ] = np.nan
  df.loc[
      df['skin_temperature_magnitude'] == 1800,
      'skin_temperature_magnitude',
  ] = np.nan
  df['skin_temperature_magnitude'] = df['skin_temperature_magnitude'] / 100
  df.loc[
      df['skin_temperature_magnitude'] > 41,
      'skin_temperature_magnitude',
  ] = 41
  return df


class Sensor(abc.ABC):

  def resample(
      timeseries_data: pd.DataFrame,
      input_timestamp_units: str = 's',
      output_timestamp_units: str = '1min',
  ) -> Any:
    """Downsamples a pandas dataframe with unknown frequency into a minutely frequency, using the column 't'.

    Args:
      timeseries_data: A pandas dataframe with a column 't' of timestamps to use
        for downsampling.
      input_timestamp_units: The units to use for the timestamps in the input
        dataframe.
      output_timestamp_units: The units to use for the timestamps in the output
        dataframe.

    Returns:
      A pandas dataframe with a minutely frequency.
    """
    # pytype: disable=wrong-arg-types
    timeseries_data['DT'] = pd.to_datetime(
        timeseries_data['t'], unit=input_timestamp_units
    )
    timeseries_data.drop(columns=['t'], inplace=True)
    timeseries_data = timeseries_data.resample(
        output_timestamp_units, on='DT'
    ).mean()
    return timeseries_data
    # pytype: enable=wrong-arg-types


class HeartRate(Sensor):
  """Heart rate sensor data."""

  sessions: list

  def __init__(
      self,
      data,
      sensor_key,
      input_timestamp_units='s',
      output_timestamp_units='1min',
  ):
    if sensor_key in data.data.keys():
      sessions = data.data[sensor_key]
      hr = []
      for session in sessions:
        times = pd.date_range(
            datetime.datetime.fromtimestamp(
                session.activity_tm.seconds, tz=datetime.timezone.utc
            ),
            periods=60 * 60 * 24,
            freq='1s',
        )
        hr_day = pd.DataFrame({'t': times, 'HR': session.bpm})
        hr_day.replace(-1, np.nan, inplace=True)
        hr.append(hr_day)
      self.hr = Sensor.resample(
          pd.concat(hr),
          input_timestamp_units='s',
          output_timestamp_units='1min',
      )
    else:
      self.hr = pd.DataFrame(columns=['DT', 'HR']).set_index('DT')
      print(ValueError(sensor_key + ' not found in data.data.keys()'))


class ContinuousEDA(Sensor):
  """Continuous EDA sensor data."""

  sessions: list

  def __init__(
      self,
      data,
      sensor_key,
      input_timestamp_units='s',
      output_timestamp_units='1min',
  ):

    if sensor_key in data.data.keys():
      sessions = data.data[sensor_key]
      continuous_eda = []
      for session in sessions:
        t = []
        for i in session.millis_from_start_time:
          t.append(
              datetime.datetime.fromtimestamp(
                  i / 1000
                  + session.activity_tm_timezone_offset * 60
                  + session.activity_tm.seconds,
                  tz=datetime.timezone.utc,
              )
          )
        times = pd.DatetimeIndex(t)
        continuous_eda_day = pd.DataFrame({
            't': times,
            'eda_level_real': session.eda_level_real,
            'eda_slope_real': session.eda_slope_real,
            'leads_contact_counts': session.leads_contact_counts,
        })
        continuous_eda_day = filter_ceda(continuous_eda_day)
        continuous_eda.append(continuous_eda_day)
      self.continuous_eda = Sensor.resample(
          pd.concat(continuous_eda),
          input_timestamp_units='s',
          output_timestamp_units='1min',
      )
    else:
      self.continuous_eda = pd.DataFrame(
          columns=[
              'DT',
              'eda_level_real',
              'eda_level_imaginary',
              'eda_slope_real',
              'eda_slope_imaginary',
              'leads_contact_counts',
          ]
      ).set_index('DT')
      print(ValueError(sensor_key + ' not found in data.data.keys()'))


class Steps(Sensor):
  """Steps sensor data."""

  sessions: list

  def __init__(
      self,
      data,
      sensor_key,
      input_timestamp_units='s',
      output_timestamp_units='1min',
  ):
    if sensor_key in data.data.keys():
      sessions = data.data[sensor_key]
      steps = []
      for session in sessions:
        times = pd.date_range(
            datetime.datetime.fromtimestamp(
                session.activity_tm.seconds, tz=datetime.timezone.utc
            ),
            periods=60 * 24,
            freq='1min',
        )
        steps_day = pd.DataFrame({'t': times, 'steps': session.steps})
        steps.append(steps_day)
      self.steps = Sensor.resample(
          pd.concat(steps),
          input_timestamp_units='s',
          output_timestamp_units='1min',
      )
    else:
      self.steps = pd.DataFrame(columns=['DT', 'steps']).set_index('DT')
      print(ValueError(sensor_key + ' not found in data.data.keys()'))


class Grok(Sensor):
  """Grok sensor data."""

  sessions: list

  def __init__(
      self,
      data,
      sensor_key,
      input_timestamp_units='s',
      output_timestamp_units='1min',
  ):

    if sensor_key in data.data.keys():
      sessions = data.data[sensor_key]
      grok = []
      for session in data.data['grok_feature_data_with_dupes']:
        t = []
        for i in session.activity_tms:
          t.append(
              datetime.datetime.fromtimestamp(
                  i.seconds, tz=datetime.timezone.utc
              )
          )
        times = pd.DatetimeIndex(t)
        grok_day = pd.DataFrame({
            't': times,
            'jerk_auto': session.jerk_auto,
            'step_count': session.step_count,
            'log_energy': session.log_energy,
            'covariance': session.covariance,
            'log_energy_ratio': session.log_energy_ratio,
            'zero_crossing_std': session.zero_crossing_std,
            'zero_crossing_avg': session.zero_crossing_avg,
            'axis_mean': session.axis_mean,
            'altim_std': session.altim_std,
            'kurtosis': session.kurtosis,
        })
        grok_day['altim_std'] = grok_day['altim_std'] / 255
        grok.append(grok_day)
      self.grok = Sensor.resample(
          pd.concat(grok),
          input_timestamp_units='s',
          output_timestamp_units='1min',
      )
    else:
      self.grok = pd.DataFrame(
          columns=[
              'DT',
              'jerk_auto',
              'step_count',
              'log_energy',
              'covariance',
              'log_energy_ratio',
              'zero_crossing_std',
              'zero_crossing_avg',
              'axis_mean',
              'altim_std',
              'kurtosis',
          ]
      ).set_index('DT')
      print(ValueError(sensor_key + ' not found in data.data.keys()'))


class SleepCoefficient(Sensor):
  """Sleep coefficient sensor data."""

  sessions: list

  def __init__(
      self,
      data,
      sensor_key,
      input_timestamp_units='s',
      output_timestamp_units='1min',
  ):
    if sensor_key in data.data.keys():
      sessions = data.data[sensor_key]
      sleep_coefficient = []
      for session in data.data['sleep_coefficient_compact']:
        times = pd.date_range(
            datetime.datetime.fromtimestamp(
                session.activity_tm.seconds, tz=datetime.timezone.utc
            ),
            periods=60 * 24 * 2,
            freq='30s',
        )
        sleep_coefficient_day = pd.DataFrame({
            't': times,
            'sleep_coefficient': session.sleep_coefficient,
            'is_on_wrist': session.is_on_wrist,
        })
        sleep_coefficient_day.loc[
            sleep_coefficient_day['sleep_coefficient'] == -1,
            'sleep_coefficient',
        ] = np.nan
        sleep_coefficient.append(sleep_coefficient_day)
      self.sleep_coefficient = Sensor.resample(
          pd.concat(sleep_coefficient),
          input_timestamp_units='s',
          output_timestamp_units='1min',
      )
    else:
      self.sleep_coefficient = pd.DataFrame(
          columns=['DT', 'sleep_coefficient', 'is_on_wrist']
      ).set_index('DT')
      print(ValueError(sensor_key + ' not found in data.data.keys()'))


class SkinTemp(Sensor):
  """Skin temperature sensor data."""

  sessions: list

  def __init__(
      self,
      data,
      sensor_key,
      input_timestamp_units='s',
      output_timestamp_units='1min',
  ):
    if sensor_key in data.data.keys():
      sessions = data.data[sensor_key]
      skin_temp = []
      for session in sessions:
        times = pd.date_range(
            datetime.datetime.fromtimestamp(
                session.activity_tm.seconds, tz=datetime.timezone.utc
            ),
            periods=60 * 24,
            freq='1min',
        )
        skintemp_day = pd.DataFrame(
            {'t': times, 'wrist_temperatures': session.wrist_temperatures}
        )
        skintemp_day = filter_wrist_temperatures(skintemp_day)
        skin_temp.append(skintemp_day)
      self.skin_temp = Sensor.resample(
          pd.concat(skin_temp),
          input_timestamp_units='s',
          output_timestamp_units='1min',
      )
    else:
      self.skin_temp = pd.DataFrame(
          columns=['DT', 'wrist_temperatures']
      ).set_index('DT')
      print(ValueError(sensor_key + ' not found in data.data.keys()'))


class MomentaryStressAlgorithm(Sensor):
  """Momentary stress algorithm sensor data."""

  sessions: list

  def __init__(
      self,
      data,
      sensor_key,
      input_timestamp_units='s',
      output_timestamp_units='1min',
  ):

    if sensor_key in data.data.keys():
      sessions = data.data[sensor_key]
      momentary_stress_algorithm = []
      for session in sessions:
        t = []
        for i in session.offsets:
          t.append(
              datetime.datetime.fromtimestamp(
                  i * 60 + session.activity_tm.seconds, tz=datetime.timezone.utc
              )
          )
        times = pd.DatetimeIndex(t)
        msa_day = pd.DataFrame({
            't': times,
            'hrv_shannon_entropy_rr': session.hrv_shannon_entropy_rr,
            'hrv_shannon_entropy_rrd': session.hrv_shannon_entropy_rrd,
            'hrv_percentage_of_nn_30': session.hrv_percentage_of_nn_30,
            'ceda_magnitude_real_micro_siemens': (
                session.ceda_magnitude_real_micro_siemens
            ),
            'ceda_slope_real_micro_siemens': (
                session.ceda_slope_real_micro_siemens
            ),
            'rmssd_percentile_0595': session.rmssd_percentile_0595,
            'sdnn_percentile_0595': session.sdnn_percentile_0595,
            'msa_probability': session.msa_probability,
            'hrv_percent_good': session.hrv_percent_good,
            'hrv_rr_80th_percentile_mean': session.hrv_rr_80th_percentile_mean,
            'hrv_rr_20th_percentile_mean': session.hrv_rr_20th_percentile_mean,
            'hrv_rr_median': session.hrv_rr_median,
            'hrv_rr_mean': session.hrv_rr_mean,
            'hr_at_rest_mean': session.hr_at_rest_mean,
            'skin_temperature_magnitude': session.skin_temperature_magnitude,
            'skin_temperature_slope': session.skin_temperature_slope,
        })
        msa_day = filter_msa(msa_day)

        momentary_stress_algorithm.append(msa_day)
      self.momentary_stress_algorithm = Sensor.resample(
          pd.concat(momentary_stress_algorithm),
          input_timestamp_units='s',
          output_timestamp_units='1min',
      )
    else:
      self.momentary_stress_algorithm = pd.DataFrame(
          columns=[
              'DT',
              'hrv_shannon_entropy_rr',
              'hrv_shannon_entropy_rrd',
              'hrv_percentage_of_nn_30',
              'ceda_magnitude_real_micro_siemens',
              'ceda_slope_real_micro_siemens',
              'rmssd_percentile_0595',
              'sdnn_percentile_0595',
              'msa_probability',
              'hrv_percent_good',
              'hrv_rr_80th_percentile_mean',
              'hrv_rr_20th_percentile_mean',
              'hrv_rr_median',
              'hrv_rr_mean',
              'hr_at_rest_mean',
              'skin_temperature_magnitude',
              'skin_temperature_slope',
          ]
      ).set_index('DT')

    print(ValueError(sensor_key + ' not found in data.data.keys()'))


@dataclasses.dataclass(frozen=True)
class ProdSession:

  # A session specific identifier for a 24hr period of data collection.
  session_id: str
  # Heart rate table data.
  hr: HeartRate
  # Continuous heart rate table data.
  continuous_eda: ContinuousEDA
  # Steps table data.
  steps: Steps
  # Grok table data.
  grok: Grok
  # Sleep Coefficient table data.
  sleep_coefficient: SleepCoefficient
  # Skin Temp table data.
  skin_temp: SkinTemp
  # MSA table data.
  momentary_stress_algorithm: MomentaryStressAlgorithm

  def join(self) -> pd.DataFrame:

    dfs = [
        self.hr,
        self.continuous_eda,
        self.steps,
        self.grok,
        self.sleep_coefficient,
        self.skin_temp,
        self.momentary_stress_algorithm,
    ]
    session = ft.reduce(
        lambda left, right: pd.merge(left, right, on='DT', how='outer'), dfs
    )
    # pytype: disable=attribute-error
    if 'is_on_wrist' in session.columns:
      session.loc[(session.is_on_wrist == 0), :] = np.nan
      session = session[sensor_constants.FEATURES_TO_INCLUDE]
      if experiment_constants.USE_NORMALIZATION:
        for feature in sensor_constants.FEATURES_TO_INCLUDE:
          session[feature] = (
              session[feature]
              - sensor_constants.NORMALIZATION_PARAMETERS[feature][0]
          ) / (sensor_constants.NORMALIZATION_PARAMETERS[feature][1])
        session = session.clip(-5, 5)
      return session
    else:
      return pd.DataFrame()
    # pytype: enable=attribute-error


def _load_user_data(
    data_key_type: str,
    user_id: str,
    data_storage_keys_to_load: list[str],
) -> data_intermediates.DataKeyAndKeyValuesWithData:
  """Loads Tier-2 user data to a DataKeyAndKeyValuesWithData.

  Args:
    data_key_type: Type of the DataKey to load.
    user_id: User ID to load.
    data_storage_keys_to_load: List of DataStorage keys that should be loaded
      for the user. All available loaded data will be in the returned
      DataKeyAndKeyValuesWithData's data field.

  Returns:
    DataKeyAndKeyValuesWithData with the loaded data.
  """
  # Each user is represented by one DataKey in the database. Each DataKey has
  # DataStorage elements associated with it. These are what will point to the
  # capacitor files with the imported raw data.
  dkkv = metadata_database_helpers.get_database_data_for_data_key(
      sensor_constants.DATABASE_PATH,
      data_key_pb2.DataKey(
          type=data_key_type,
          session_id=user_id,
      ),
  )

  dkkvwd = list(
      data_loading.LoadDataDoFn(
          data_storage_keys_to_load, data_loader.get_data_loader()
      ).process(dkkv)
  )[0]
  return dkkvwd


# Loop through users
def window(
    ids: list[str],
    window_length: str,
    timestamp_units: str,
    num_of_sensor_features: int,
) -> Any:
  """Window the data into windows."""
  window_length_int = int(window_length.split('m')[0])
  for user_id in ids:
    dkkvwd = _load_user_data(
        data_key_type=sensor_constants.DATA_KEY_TYPE,
        user_id=user_id,
        data_storage_keys_to_load=sensor_constants.DATA_STORAGE_KEYS_TO_LOAD,
    )

    momentary_stress_algorithm_loader = (
        common_data_helper.get_fitbit_prod_data_types_to_loaders()[
            constants.DataStorageType.MOMENTARY_STRESS_ALGORITHM.value
        ]
    )
    dkkvwd.data['momentary_stress_algorithm_compact'] = (
        momentary_stress_algorithm_loader(
            dkkvwd.data_key_and_key_values.data_storage_dict[
                'momentary_stress_algorithm'
            ].storage_path
        )
    )

    wrist_temperature_loader = (
        common_data_helper.get_fitbit_prod_data_types_to_loaders()[
            constants.DataStorageType.WRIST_TEMPERATURE.value
        ]
    )
    dkkvwd.data['wrist_temperature'] = wrist_temperature_loader(
        dkkvwd.data_key_and_key_values.data_storage_dict[
            'wrist_temperature'
        ].storage_path
    )

    sess = ProdSession(
        session_id=user_id,
        hr=HeartRate(
            dkkvwd,
            'heart_rate_with_dupes',
            input_timestamp_units='s',
            output_timestamp_units='1min',
        ).hr,
        continuous_eda=ContinuousEDA(
            dkkvwd,
            'continuous_eda',
            input_timestamp_units='s',
            output_timestamp_units='1min',
        ).continuous_eda,
        steps=Steps(
            dkkvwd,
            'steps_compact',
            input_timestamp_units='s',
            output_timestamp_units='1min',
        ).steps,
        grok=Grok(
            dkkvwd,
            'grok_feature_data_with_dupes',
            input_timestamp_units='s',
            output_timestamp_units='1min',
        ).grok,
        sleep_coefficient=SleepCoefficient(
            dkkvwd,
            'sleep_coefficient_compact',
            input_timestamp_units='s',
            output_timestamp_units='1min',
        ).sleep_coefficient,
        skin_temp=SkinTemp(
            dkkvwd,
            'wrist_temperature',
            input_timestamp_units='s',
            output_timestamp_units='1min',
        ).skin_temp,
        momentary_stress_algorithm=MomentaryStressAlgorithm(
            dkkvwd,
            'momentary_stress_algorithm_compact',
            input_timestamp_units='s',
            output_timestamp_units='1min',
        ).momentary_stress_algorithm,
    )

    sess_joined = sess.join()

    df_grouped = sess_joined.groupby(pd.Grouper(freq=window_length))
    for name, data in df_grouped:
      data = data.apply(lambda col: pd.to_numeric(col))
      if not experiment_constants.DOUBLE_THRESHOLD:
        signal, nan_mask, missingness_ratio = process_each_window(
            data, experiment_constants.MISSING_RATIO_THRESHOLD
        )
      else:
        signal, nan_mask, missingness_ratio = (
            process_each_window_with_double_threshold(
                data, experiment_constants.MISSING_RATIO_THRESHOLD
            )
        )
      if signal == 'BREAK':
        break
      elif signal == 'CONTINUE':
        continue
      elif signal == 'PROCESS':
        data = data.interpolate(method='time').bfill().fillna(0).to_numpy()
        if (
            data.shape[0] == window_length_int
            and data.shape[1] == num_of_sensor_features
        ):
          yield name, {
              lsm_constants.TFExampleKey.INPUT.value: data,
              lsm_constants.TFExampleKey.MASK.value: nan_mask,
              lsm_constants.TFExampleKey.MISSINGNESS_RATIO.value: float(
                  missingness_ratio
              ),
              lsm_constants.TFExampleKey.USER_ID.value: user_id,
              lsm_constants.TFExampleKey.KEY.value: str(name),
              lsm_constants.TFExampleKey.CAPTION.value: (
                  sensors_caption.generate_caption(data, nan_mask)
              ),
          }
      else:
        raise ValueError('Unknown signal: %s' % signal)
