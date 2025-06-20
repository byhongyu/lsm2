"""LSM Dataset Constants.

This file contains known / calculated constants for used datasets.
This file is needed as these constants are often unavailable at runtime. For
example, the cardinality of a filtered TF dataset cannot be determined without
a full pass of the dataset. Additionally, dataset in a non-TFDS format do not
have associated metadatas.

Fields for each dataset may include:
  - dataset_name: The name of the dataset.
  - num_train_examples: The number of training examples in the dataset.
  - num_test_examples: The number of test examples in the dataset.
  - feature_shape: The shape of the input features in the dataset.
  - datetime_features: A dictionary of datetime feature names and indices.
  - label_values: The possible numeric label values in the dataset.
  - label_value_offset: The numeric offset for the label values.
  - label_names: The string names of the label values.
  - label_counts: The number of samples for each label.
  - label_weights: The weights for each label.
  - filter_log_values: Whether to filter log values.
  - total_label_values: The total number of pre-filtered class labels.

  TODO(girishvn, xumax): Create seperate configs for each dataset. And convert
  this file to a class config format. Additionally, improve the format of
  dataset constants. E.g. use a dictionary for label values, names, counts, etc.

  NOTE: LSMv1 datasets, used for ICLR '25 experiments, have been deprecated, and
  are no longer supported. Check file history for the deprecated dataset
  constants, and datasets/deprecated_datasets/ for the datasets themselves.
"""

from typing import Any, Dict

# Data constants
MINUTES_IN_ONE_DAY = 1440
MINUTES_IN_ONE_WEEK = 10080
NUM_SENSOR_FEATS_LSM_V2 = 26  # number of sensor features in the V2 dataset.

lsm_dataset_constants: Dict[str, Any] = {
    ###############################
    # V2  Pre-training Datasets for Post-NeurIPS 2025
    ###############################
    # limit pre-training to days with activity
    'SensorLMDatasetActivity_NeurIPS2025_Tier2Prod_train_only_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.8_timestamp_202506020400_doublethreshold_False_usetier2label_True': {
        'dataset_name': (
            'SensorLMDatasetActivity_NeurIPS2025_Tier2Prod_train_only_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.8_timestamp_202506020400_doublethreshold_False_usetier2label_True'
        ),
        'num_train_examples': 1638768,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    ###############################
    # V2  Pre-training Datasets for Metabolic Health Nature Paper
    # The future V2 dataset might include datetime features.
    ###############################
    'HomeIR_Nature_lsm_v2_train_only_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.8_timestamp_202505202228_doublethreshold_False': {
        'dataset_name': (
            'HomeIR_Nature_lsm_v2_train_only_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.8_timestamp_202505202228_doublethreshold_False'
        ),
        'num_train_examples': 21691434,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    ###############################
    # V2  Pre-training Datasets for NeurIPS 2025
    # The future V2 dataset might include datetime features.
    ###############################
    'lsm_v2_train_only_sessions_10000_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504172035_doublethreshold_False': {
        'dataset_name': (
            'lsm_v2_train_only_sessions_10000_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504172035_doublethreshold_False'
        ),
        'num_train_examples': 282244,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    'lsm_v2_train_only_sessions_1000_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504171841_doublethreshold_False': {
        'dataset_name': (
            'lsm_v2_train_only_sessions_1000_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504171841_doublethreshold_False'
        ),
        'num_train_examples': 26268,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    'lsm_v2_train_only_sessions_100_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504171758_doublethreshold_False': {
        'dataset_name': (
            'lsm_v2_train_only_sessions_100_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504171758_doublethreshold_False'
        ),
        'num_train_examples': 2308,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },

    'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.2_timestamp_202504110407_doublethreshold_False': {
        'dataset_name': (
            'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.2_timestamp_202504110407_doublethreshold_False'
        ),
        'num_train_examples': 12429,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504110538_doublethreshold_False': {
        'dataset_name': (
            'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.5_timestamp_202504110538_doublethreshold_False'
        ),
        'num_train_examples': 1901088,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.8_timestamp_202504110551_doublethreshold_False': {
        'dataset_name': (
            'lsm_v2_pretraining_sessions_-1_windowsize_1440_sensorfeatures_26_validonly_False_missingratio_0.8_timestamp_202504110551_doublethreshold_False'
        ),
        'num_train_examples': 3581748,
        'num_test_examples': 0,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    'lsm_v2_missing_balanced_20250301_valid_dataset': {
        'dataset_name': 'lsm_v2_missing_balanced_20250301_valid_dataset',
        'num_train_examples': 0,
        'num_test_examples': 6000,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    'lsm_v2_missing_balanced_20250301_valid_dataset_bounded_50p': {
        'dataset_name': (
            'lsm_v2_missing_balanced_20250301_valid_dataset_bounded_50p'
        ),
        'num_train_examples': 0,
        'num_test_examples': 6000,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    'lsm_v2_missing_balanced_20250502_valid_dataset_bounded_30p': {
        'dataset_name': (
            'lsm_v2_missing_balanced_20250502_valid_dataset_bounded_30p'
        ),
        'num_train_examples': 0,
        'num_test_examples': 6000,
        'feature_shape': (
            MINUTES_IN_ONE_DAY,
            NUM_SENSOR_FEATS_LSM_V2,
            1,
        ),
    },
    ###############################
    # V2  Downstream Task Datasets
    ###############################
    # Metabolic Health Dataset
    # Derived from the chr-ards-metabolichealth-deid data sandbox.
    # a. 10080m (weekly)
    # TODO(girishvn): onboard this dataset.
    # b. 1440m (daily)
    # Previously sourced from:
    # '/namespace/fitbit-medical-sandboxes/partner/encrypted/chr-ards-metabolichealth/deid/exp/aliheydari/'
    # Now sourced from:
    # '/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-metabolichealth/deid/exp/aliheydari/'
    'metabolic_tfrecords_24h_missingness_80': {
        'dataset_name': 'metabolic_tfrecords_24h_missingness_80',
        'dataset_dir': (
            '/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-metabolichealth/deid/exp/aliheydari/'
        ),
        'num_train_examples': 54385,
        'num_test_examples': 10664,
        'feature_shape': (MINUTES_IN_ONE_DAY, NUM_SENSOR_FEATS_LSM_V2, 1),
        'data_split_id': 70467,
        # Task details
        'homa_ir_binray': {
            'label_values': [0, 1],
            'label_names': ['Negative', 'Positive'],
            'label_counts': [48020, 6365],
            'label_weights': None,
        },
        'hypertension_binary': {
            'label_values': [0, 1],
            'label_names': ['Negative', 'Positive'],
            'label_counts': [41843, 12542],
            'label_weights': None,
        },
        'hyperlipidemia_binary': {
            'label_values': [0, 1],
            'label_names': ['Negative', 'Positive'],
            'label_counts': [43576, 10809],
            'label_weights': None,
        },
        'diabetes_binary': {
            'label_values': [0, 1],
            'label_names': ['Negative', 'Positive'],
            'label_counts': [49434, 4951],
            'label_weights': None,
        },
        'anxiety_binary': {
            'label_values': [0, 1],
            'label_names': ['Negative', 'Positive'],
            'label_counts': [34185, 20200],
            'label_weights': None,
        },
        'cardiovascular_binary': {
            'label_values': [0, 1],
            'label_names': ['Negative', 'Positive'],
            'label_counts': [52994, 1391],
            'label_weights': None,
        },
        'respiratory': {
            'label_values': [0, 1],
            'label_names': ['Negative', 'Positive'],
            'label_counts': [46095, 8290],
            'label_weights': None,
        },
    },
    ###############################
    # V1  Pre-training Datasets:
    ###############################
    # Pretrain V3 (8M) Dataset: This dataset will be used for model
    # pretraining. It is a combination of adding the ENTIRE V1 pretraining test
    # (both splits of lsm_300min_10M_impute) to the train split of the V2
    # pretraining set (lsm_300min_pretraining_165K_n10).
    'lsm_300min_pretraining_8M_combined': {
        'dataset_name': 'lsm_300min_pretraining_8M_combined',
        'num_train_examples': 7867175,
        'num_test_examples': 330694,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
    },
    # Pretrain V2 (2M) Dataset: This dataset will be used for model
    # pretraining. It is randomly sampled and does NOT neccessarily mood /
    # exercise / stress event, as it true with the other datasets. This replaces
    # the V1 dataset above.
    'lsm_300min_pretraining_165K_n10': {
        'dataset_name': 'lsm_300min_pretraining_165K_n10',
        'num_train_examples': 1321235,
        'num_test_examples': 330694,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
    },
    # Subject Scaling Pretraining Datasets:
    # 100 Users
    'lsm_300min_pretraining_100_n10_unshuffled': {
        'dataset_name': 'lsm_300min_pretraining_100_n10_unshuffled',
        'num_train_examples': 1010,
        'num_test_examples': 330694,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
    },
    # 1K Users
    'lsm_300min_pretraining_1K_n10_unshuffled': {
        'dataset_name': 'lsm_300min_pretraining_1K_n10_unshuffled',
        'num_train_examples': 9930,
        'num_test_examples': 330694,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
    },
    # 10K Users
    'lsm_300min_pretraining_10K_n10_unshuffled': {
        'dataset_name': 'lsm_300min_pretraining_10K_n10_unshuffled',
        'num_train_examples': 102331,
        'num_test_examples': 330694,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
    },
    # 75K Users
    'lsm_300min_pretraining_75K_n10_unshuffled': {
        'dataset_name': 'lsm_300min_pretraining_75K_n10_unshuffled',
        'num_train_examples': 763681,
        'num_test_examples': 330694,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
    },
    ###############################
    # Downstream Task Datasets:
    ###############################
    # 10 Class Activity (16K) Balanced Dataset:
    'lsm_300min_600_activities_balanced_v4': {
        'dataset_name': 'lsm_300min_600_activities_balanced_v4',
        'num_train_examples': 14372,
        'num_test_examples': 3262,
        'total_num_train_examples': 15146,
        'total_num_test_examples': 3320,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
        'label_values': [
            91043,
            90024,
            90017,
            90013,
            90009,
            90001,
            90012,
            90019,
            91040,
            91042,
        ],
        'label_names': [
            'Weightlifting',
            'Swim',
            'Elliptical',
            'Walk',
            'Run',
            'Bike',
            'Hike',
            'Treadmill',
            'HIIT',
            'Strength training',
        ],
        # NOTE: These weights were calculated such that for a class i:
        # class_i_weight = total_samples / (num_classes * class_i_samples)
        'label_weights': [
            2.1482810164424513,
            0.616295025728988,
            9.455263157894738,
            0.20868302599099753,
            0.7726881720430108,
            1.2067170445004198,
            2.561853832442068,
            9.038993710691823,
            4.328915662650602,
            6.275982532751092,
        ],
        # Samples per class
        'label_counts': [669, 2332, 152, 6887, 1860, 1191, 561, 159, 332, 229],
        'label_value_offset': 65536,
        'filter_log_values': True,
        # Total number of pre-filtered class labels.
        'total_label_values': [
            52000,
            91043,
            90024,
            90017,
            90013,
            90009,
            56001,
            90005,
            90001,
            90012,
            90019,
            91046,
            91040,
            90014,
            91042,
            91057,
            53000,
            55001,
        ],
    },
    # 8 Class (Remapped) Activity (16K) Balanced Dataset:
    'lsm_300min_600_activities_remapped_8class': {
        'dataset_name': 'lsm_300min_600_activities_balanced_v4',
        'num_train_examples': 14372,
        'num_test_examples': 3262,
        'total_num_train_examples': 15146,
        'total_num_test_examples': 3320,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
        'label_values': [
            91043,
            90024,
            90017,
            90013,
            90009,
            90001,
            91040,
            91042,
        ],
        'label_names': [
            'Weightlifting',
            'Swim',
            'Elliptical',
            'Walk',
            'Run',
            'Bike',
            'HIIT',
            'Strength training',
        ],
        # NOTE: These weights were calculated such that for a class i:
        # class_i_weight = total_samples / (num_classes * class_i_samples)
        'label_weights': [
            2.685,
            0.770,
            11.819,
            0.236,
            0.966,
            1.508,
            5.411,
            7.845,
        ],
        # Samples per class
        'label_counts': [669, 2332, 152, 7607, 1860, 1191, 332, 229],
        'label_value_offset': 65536,
        'filter_log_values': True,
        # Total number of pre-filtered class labels.
        'total_label_values': [
            52000,
            91043,
            90024,
            90017,
            90013,
            90009,
            56001,
            90005,
            90001,
            90012,
            90019,
            91046,
            91040,
            90014,
            91042,
            91057,
            53000,
            55001,
        ],
    },
    # FEWSHOT 8 Class (Remapped) Activity (16K) Balanced Dataset:
    'fewshot_lsm_300min_600_activities_remapped_8class': {
        'dataset_name': 'lsm_300min_600_activities_balanced_v4',
        'num_train_examples': None,
        'num_test_examples': 3262,
        'total_num_train_examples': 15146,
        'total_num_test_examples': 3320,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
        'label_values': [
            91043,
            90024,
            90017,
            90013,
            90009,
            90001,
            91040,
            91042,
        ],
        'label_names': [
            'Weightlifting',
            'Swim',
            'Elliptical',
            'Walk',
            'Run',
            'Bike',
            'HIIT',
            'Strength training',
        ],
        # NOTE: These weights were calculated such that for a class i:
        # class_i_weight = total_samples / (num_classes * class_i_samples)
        'label_weights': None,
        # Samples per class
        'label_counts': [669, 2332, 152, 7607, 1860, 1191, 332, 229],
        'label_value_offset': 65536,
        'filter_log_values': True,
        # Total number of pre-filtered class labels.
        'total_label_values': [
            52000,
            91043,
            90024,
            90017,
            90013,
            90009,
            56001,
            90005,
            90001,
            90012,
            90019,
            91046,
            91040,
            90014,
            91042,
            91057,
            53000,
            55001,
        ],
    },
    # Biological Sex 2 Class (Derived from Activity Dataset):
    'lsm_300min_600_biological_sex': {
        'dataset_name': 'lsm_300min_600_activities_balanced_v4',
        'num_train_examples': 14203,
        'num_test_examples': 3250,
        'total_num_train_examples': 15146,
        'total_num_test_examples': 3320,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
        'label_values': [1, 2],
        'label_names': ['Male', 'Female'],
        'label_counts': [5022, 9181],  # Samples per class
        'label_value_offset': 0,
        'filter_log_values': True,
        # Total number of pre-filtered class labels.
        'total_label_values': [0, 1, 2],
    },
    # Binned Age 4 Class (Derived from Activity Dataset):
    'lsm_300min_600_binnned_age': {
        'dataset_name': 'lsm_300min_600_activities_balanced_v4',
        'num_train_examples': 14372,
        'num_test_examples': 3262,
        'total_num_train_examples': 15146,
        'total_num_test_examples': 3320,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
        'label_values': [0, 1, 2, 3],
        'label_names': ['18-34', '35-49', '50-64', '65+'],
        'label_counts': [1179, 5697, 5246, 2250],  # Samples per class
        'label_value_offset': 0,
        'filter_log_values': True,
        # Total number of pre-filtered class labels.
        'total_label_values': [0, 1, 2, 3],
    },
    # Mood (7K) Balanced Dataset:
    # This dataset is balanced across 5 mood events.
    'lsm_300min_2000_mood_balanced': {
        'dataset_name': 'lsm_300min_2000_mood_balanced',
        'num_train_examples': 6195,
        'num_test_examples': 1329,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
        'label_values': [65543, 65546, 65547, 65548, 65549],
        'label_value_offset': 65536,
        'label_names': [
            'Content',
            'Frustrated',
            'Excited',
            'Calm',
            'Stressed',
        ],
        'label_counts': [1109, 1113, 1289, 1092, 1592],
    },
    # Stress (7K) Balanced  Dataset:
    # This dataset is balanced across a binary stress event.
    # NOTE: This dataset is a copy of the the above mood dataset, but uses a
    # binary stress label, instead of a multi-class mood label.
    'lsm_300min_2000_stress_balanced': {
        'dataset_name': 'lsm_300min_2000_stress_balanced',
        'num_train_examples': 6195,
        'num_test_examples': 1329,
        'datetime_features': {
            'names': ['min_of_hr', 'hr_of_day', 'day_of_week', 'month_of_year'],
            'indices': [26, 27, 28, 29],
        },
    },
}
