#!/bin/bash
set -e
source <(python3 medical/waveforms/modelling/lsm/datasets/sensorlm/experiment_constants.py)
echo "WINDOW_SIZE: $WINDOW_SIZE"
echo "NUMBER_OF_SESSIONS: $NUMBER_OF_SESSIONS"
echo "NUMBER_OF_SENSOR_FEATURES: $NUMBER_OF_SENSOR_FEATURES"
echo "VALID_ONLY: $VALID_ONLY"
echo "MISSING_RATIO_THRESHOLD: $MISSING_RATIO_THRESHOLD"
echo "TIMESTAMP: $TIMESTAMP"
echo "DOUBLE_THRESHOLD: $DOUBLE_THRESHOLD"
echo "DATA_TYPE: $DATA_TYPE"

DATASET_NAME="sensorlm_${DATA_TYPE}_sessions_${NUMBER_OF_SESSIONS}_winsize_${WINDOW_SIZE}_feats_${NUMBER_OF_SENSOR_FEATURES}_validonly_${VALID_ONLY}_missingratio_${MISSING_RATIO_THRESHOLD}_${TIMESTAMP}"
echo "DATASET_NAME: $DATASET_NAME" #optional, to see the value
rabbit --verifiable mpm --stamp -c opt \
medical/waveforms/modelling/lsm/datasets/sensorlm:download_and_prepare_mpm \
--mpm_build_arg=--label="dev-$USER" \
--mpm_build_arg=--durability=ephemeral

borgcfg medical/waveforms/modelling/lsm/datasets/sensorlm/download_and_prepare.borg up --skip_confirmation --vars=datasets=sensorlm,sub_cns_dir="$DATASET_NAME"