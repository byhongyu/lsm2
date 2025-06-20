import numpy as np
import pandas as pd

from google3.medical.waveforms.modelling.lsm.datasets.lsm import sensors
from google3.testing.pybase import googletest


class IsNanPercentWithinRangeTest(googletest.TestCase):

  def test_within_range(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    self.assertTrue(sensors.is_nan_percent_within_range(data, 0.2, 0.3))

  def test_outside_range_low(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    self.assertFalse(sensors.is_nan_percent_within_range(data, 0.3, 0.4))

  def test_outside_range_high(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    self.assertFalse(sensors.is_nan_percent_within_range(data, 0.1, 0.2))

  def test_empty_dataframe(self):
    self.assertFalse(
        sensors.is_nan_percent_within_range(pd.DataFrame(), 0.1, 0.4)
    )

  def test_all_nan_dataframe(self):
    data = pd.DataFrame({'col1': [np.nan, np.nan], 'col2': [np.nan, np.nan]})
    self.assertFalse(sensors.is_nan_percent_within_range(data, 0.1, 0.4))

  def test_zero_total_values(self):
    data = pd.DataFrame({'col1': [], 'col2': []})
    self.assertFalse(sensors.is_nan_percent_within_range(data, 0.1, 0.4))


class CheckIfBreakTest(googletest.TestCase):

  def test_below_threshold(self):
    dict_count = {(0.9, 0.8): 5, (0.8, 0.7): 3}
    self.assertFalse(sensors.check_if_break(dict_count, 4))

  def test_above_threshold(self):
    dict_count = {(0.9, 0.8): 5, (0.8, 0.7): 5}
    self.assertTrue(sensors.check_if_break(dict_count, 4))

  def test_empty_dict_count(self):
    self.assertTrue(sensors.check_if_break({}, 4))


class CheckIfSkipTest(googletest.TestCase):

  def test_skip_condition_met(self):
    missing_ratio = 0.85
    dict_count = {(0.9, 0.8): 10}
    max_per_user_window_size = 10
    self.assertTrue(
        sensors.check_if_skip(
            missing_ratio, dict_count, max_per_user_window_size
        )
    )

  def test_skip_condition_not_met_ratio(self):
    missing_ratio = 0.75
    dict_count = {(0.9, 0.8): 10}
    max_per_user_window_size = 10
    self.assertFalse(
        sensors.check_if_skip(
            missing_ratio, dict_count, max_per_user_window_size
        )
    )

  def test_skip_condition_not_met_count(self):
    missing_ratio = 0.85
    dict_count = {(0.9, 0.8): 5}
    max_per_user_window_size = 10
    self.assertFalse(
        sensors.check_if_skip(
            missing_ratio, dict_count, max_per_user_window_size
        )
    )

  def test_empty_dict_count(self):
    self.assertFalse(sensors.check_if_skip(0.85, {}, 10))


class IsNanPercentAboveTest(googletest.TestCase):

  def test_above_threshold(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    self.assertTrue(sensors._is_nan_percent_above(data, 0.2))

  def test_below_threshold(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    self.assertFalse(sensors._is_nan_percent_above(data, 0.3))

  def test_empty_dataframe(self):
    self.assertFalse(sensors._is_nan_percent_above(pd.DataFrame(), 0.1))

  def test_all_nan_dataframe(self):
    data = pd.DataFrame({'col1': [np.nan, np.nan], 'col2': [np.nan, np.nan]})
    self.assertTrue(sensors._is_nan_percent_above(data, 0.1))

  def test_zero_total_values(self):
    data = pd.DataFrame({'col1': [], 'col2': []})
    self.assertFalse(sensors._is_nan_percent_above(data, 0.1))


class GenerateMissingMaskRatioTest(googletest.TestCase):

  def test_generate_mask(self):
    data = pd.DataFrame({'col1': [1, 4, 2, np.nan], 'col2': [4, 2, np.nan, 6]})
    mask, ratio = sensors._generate_missing_mask_ratio(data)
    expected_mask = np.array(
        [[False, False], [False, False], [False, True], [True, False]]
    )
    self.assertTrue(np.array_equal(mask, expected_mask))
    self.assertEqual(ratio, 0.25)

  def test_all_nan_array(self):
    mask, ratio = sensors._generate_missing_mask_ratio(
        pd.DataFrame({'col1': [np.nan, np.nan]})
    )
    self.assertTrue(np.array_equal(mask, np.array([[True], [True]])))
    self.assertEqual(ratio, 1.0)


class ProcessEachWindowTest(googletest.TestCase):

  def test_process_window(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    status, mask, ratio = sensors.process_each_window(data, 0.3)
    self.assertEqual(status, 'PROCESS')
    self.assertIsNotNone(mask)
    self.assertEqual(ratio, 0.25)

  def test_skip_window_high_nan(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    status, _, _ = sensors.process_each_window(data, 0.1)
    self.assertEqual(status, 'CONTINUE')

  def test_empty_dataframe(self):
    status, mask, ratio = sensors.process_each_window(pd.DataFrame(), 0.3)
    self.assertEqual(status, 'CONTINUE')
    self.assertIsNone(mask)
    self.assertIsNone(ratio)


class ProcessEachWindowWithDoubleThresholdTest(googletest.TestCase):

  def test_empty_dataframe(self):
    status, mask, ratio = sensors.process_each_window_with_double_threshold(
        pd.DataFrame(), 0.3
    )
    self.assertEqual(status, 'CONTINUE')
    self.assertIsNone(mask)
    self.assertIsNone(ratio)

  def test_missingness_ratio_within_double_threshold(self):
    data = pd.DataFrame({'col1': [1, 2, np.nan, 4], 'col2': [5, np.nan, 7, 8]})
    status, mask, ratio = sensors.process_each_window_with_double_threshold(
        data, 0.3
    )
    self.assertEqual(status, 'PROCESS')
    self.assertIsNotNone(mask)
    self.assertAlmostEqual(ratio, 0.25)

  def test_missingness_ratio_outside_double_threshold(self):
    data = pd.DataFrame(
        {'col1': [1, 2, np.nan, 4, np.nan], 'col2': [5, np.nan, 7, 8, np.nan]}
    )
    status, mask, ratio = sensors.process_each_window_with_double_threshold(
        data, 0.3
    )
    self.assertEqual(status, 'CONTINUE')
    self.assertIsNone(mask)
    self.assertIsNone(ratio)


class TestProcessEachWindowWithStratifiedUsers(googletest.TestCase):

  def test_empty_dataframe(self):
    data = pd.DataFrame()
    sensor_window_count_dict = {(0.1, 0.0): 0, (0.2, 0.1): 0}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.5
    result = sensors.process_each_window_with_stratified_users(
        data,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )
    self.assertEqual(result, ('CONTINUE', None, None))

  def test_break_condition(self):
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    sensor_window_count_dict = {(0.1, 0.0): 2, (0.2, 0.1): 2}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.5
    result = sensors.process_each_window_with_stratified_users(
        data,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )
    self.assertEqual(result, ('BREAK', None, None))

  def test_high_nan_percentage(self):
    data = pd.DataFrame({'col1': [1, np.nan], 'col2': [np.nan, np.nan]})
    sensor_window_count_dict = {(0.1, 0.0): 0, (0.2, 0.1): 0}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.5
    result = sensors.process_each_window_with_stratified_users(
        data,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )
    self.assertEqual(result, ('CONTINUE', None, None))

  def test_skip_condition(self):
    data = pd.DataFrame({'col1': [1, np.nan], 'col2': [3, 4]})  # 25% NaN
    sensor_window_count_dict = {(0.3, 0.2): 2, (0.4, 0.3): 0}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.1
    result = sensors.process_each_window_with_stratified_users(
        data,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )
    self.assertEqual(result, ('CONTINUE', None, None))

  def test_process_condition(self):
    data = pd.DataFrame({'col1': [1, np.nan], 'col2': [3, 4]})  # 25% NaN
    sensor_window_count_dict = {(0.3, 0.2): 0, (0.4, 0.3): 0}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.5
    result = sensors.process_each_window_with_stratified_users(
        data,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )
    self.assertEqual(result[0], 'PROCESS')
    self.assertEqual(sensor_window_count_dict, {(0.3, 0.2): 1, (0.4, 0.3): 0})

  def test_process_condition_multiple_updates(self):
    data1 = pd.DataFrame({'col1': [1, np.nan], 'col2': [3, 4]})  # 25% NaN
    data2 = pd.DataFrame({'col1': [1, np.nan], 'col2': [np.nan, 4]})  # 50% NaN
    sensor_window_count_dict = {(0.3, 0.2): 0, (0.6, 0.5): 0}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.7

    result1 = sensors.process_each_window_with_stratified_users(
        data1,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )
    result2 = sensors.process_each_window_with_stratified_users(
        data2,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )

    self.assertEqual(result1[0], 'PROCESS')
    self.assertEqual(result2[0], 'PROCESS')
    self.assertEqual(sensor_window_count_dict, {(0.3, 0.2): 1, (0.6, 0.5): 1})

  def test_process_condition_edge_case_lower_bound(self):
    data = pd.DataFrame({'col1': [1, np.nan], 'col2': [3, 4]})  # 25% NaN
    sensor_window_count_dict = {(0.3, 0.25): 0, (0.4, 0.3): 0}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.5

    result = sensors.process_each_window_with_stratified_users(
        data,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )

    self.assertEqual(result[0], 'PROCESS')
    # Check count updated correctly (lower bound inclusive)
    self.assertEqual(sensor_window_count_dict, {(0.3, 0.25): 1, (0.4, 0.3): 0})

  def test_process_condition_edge_case_upper_bound(self):
    data = pd.DataFrame({'col1': [1, np.nan], 'col2': [3, 4]})  # 25% NaN
    sensor_window_count_dict = {(0.25, 0.2): 0, (0.4, 0.3): 0}
    max_per_user_window_size = 2
    missingness_ratio_threshold = 0.5

    result = sensors.process_each_window_with_stratified_users(
        data,
        sensor_window_count_dict,
        max_per_user_window_size,
        missingness_ratio_threshold,
    )

    self.assertEqual(result[0], 'PROCESS')
    # Check count updated correctly (upper bound exclusive)
    self.assertEqual(sensor_window_count_dict, {(0.25, 0.2): 0, (0.4, 0.3): 0})


if __name__ == '__main__':
  googletest.main()
