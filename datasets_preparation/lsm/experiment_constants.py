"""Constants for the experiment."""

import datetime

TRAIN_ID_CSV = "/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/session_ids/lsm_neurips_train_ids.csv"
VALID_ID_CSV = "/namespace/fitbit-medical-sandboxes/jg/partner/encrypted/chr-ards-fitbit-prod-research/deid/exp/dmcduff/ttl=52w/session_ids/lsm_neurips_valid_ids.csv"
WINDOW_SIZE = 1440  # 1440 minutes = 1 day, 10080 minutes = 1 week.
NUMBER_OF_SESSIONS = -1  # -1 for all sessions
NUMBER_OF_SENSOR_FEATURES = 26
VALID_ONLY = False
TRAIN_ONLY = True
MISSING_RATIO_THRESHOLD = 0.5
USE_NORMALIZATION = True
DOUBLE_THRESHOLD = False
if VALID_ONLY:
  DATA_TYPE = "valid_only"
elif TRAIN_ONLY:
  DATA_TYPE = "train_only"
else:
  DATA_TYPE = "train_and_valid"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M")

print(f"WINDOW_SIZE='{WINDOW_SIZE}'")
print(f"NUMBER_OF_SESSIONS='{NUMBER_OF_SESSIONS}'")
print(f"NUMBER_OF_SENSOR_FEATURES='{NUMBER_OF_SENSOR_FEATURES}'")
print(f"VALID_ONLY='{VALID_ONLY}'")
print(f"MISSING_RATIO_THRESHOLD='{MISSING_RATIO_THRESHOLD}'")
print(f"USE_NORMALIZATION='{USE_NORMALIZATION}'")
print(f"TIMESTAMP='{TIMESTAMP}'")
print(f"DOUBLE_THRESHOLD='{DOUBLE_THRESHOLD}'")
print(f"DATA_TYPE='{DATA_TYPE}'")
