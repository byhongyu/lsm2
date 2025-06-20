#!/bin/bash
set -e
source <(python3 medical/waveforms/modelling/lsm/datasets/lsm/experiment_constants.py)
echo "WINDOW_SIZE: $WINDOW_SIZE"
echo "NUMBER_OF_SESSIONS: $NUMBER_OF_SESSIONS"
echo "NUMBER_OF_SENSOR_FEATURES: $NUMBER_OF_SENSOR_FEATURES"
echo "VALID_ONLY: $VALID_ONLY"
echo "MISSING_RATIO_THRESHOLD: $MISSING_RATIO_THRESHOLD"
echo "TIMESTAMP: $TIMESTAMP"
echo "DOUBLE_THRESHOLD: $DOUBLE_THRESHOLD"
echo "DATA_TYPE: $DATA_TYPE"

DATASET_NAME="lsm_v2_${DATA_TYPE}_sessions_${NUMBER_OF_SESSIONS}_windowsize_${WINDOW_SIZE}_sensorfeatures_${NUMBER_OF_SENSOR_FEATURES}_validonly_${VALID_ONLY}_missingratio_${MISSING_RATIO_THRESHOLD}_timestamp_${TIMESTAMP}_doublethreshold_${DOUBLE_THRESHOLD}"
echo "DATASET_NAME: $DATASET_NAME" #optional, to see the value
rabbit --verifiable mpm --stamp -c opt \
medical/waveforms/modelling/lsm/datasets/lsm:download_and_prepare_mpm \
--mpm_build_arg=--label="dev-$USER" \
--mpm_build_arg=--durability=ephemeral

borgcfg medical/waveforms/modelling/lsm/datasets/lsm/download_and_prepare.borg up --skip_confirmation --vars=datasets=lsm,sub_cns_dir="$DATASET_NAME"