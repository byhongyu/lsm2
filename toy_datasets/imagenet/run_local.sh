#!/bin/bash

source gbash.sh || exit

DEFINE_string config 'default.py' \
  'Config file (and optional `arg`) to run.'
DEFINE_string workdir \
  "/tmp/jax_imagenet_$(date +%Y%m%d_%H%M%S)" \
  'Directory for storing model outputs'
DEFINE_string uptc 'uptc://prod/jellyfish_2x2' \
  'UPTC pool to run the program against.'
DEFINE_bool ml_python false \
  'If true use ml_python to run train.py, otherwise run the build target.'

readonly GOOGLE3="$(gbash::get_google3_dir)"
readonly SCRIPT_PATH="$(realpath "$0")"
readonly MODEL_BASE="$(dirname "${SCRIPT_PATH}" | sed -e "s%${GOOGLE3}/%%")"
readonly CONFIG_BASE="${MODEL_BASE}/configs"

# Allow passing through additional parameters to the training script.
# This can be used to overwrite config values.
GBASH_PASSTHROUGH_UNKNOWN_FLAGS=1

ADHOC_IMPORT_MODULES="google3.experimental.largesensormodels.toy_datasets.imagenet"
ADHOC_IMPORT_MODULES+=",google3.third_party.py.clu"

function main() {
  set -e
  cd "$(gbash::get_google3_dir)"

  args=(--alsologtostderr)
  args+=(--workdir="${FLAGS_workdir}")

  if [[ ! -z "${FLAGS_uptc}" ]]; then
    args+=(--jax_backend_target="${FLAGS_uptc}")
    args+=(--jax_xla_backend="pathways")
  fi

  args+=(--config="$(pwd)/${CONFIG_BASE}/${FLAGS_config}")
  # Additional flags at the end. This is important to allow overwriting values
  # in the config.
  args+=("${GBASH_ARGV[@]}")

  if (( FLAGS_ml_python )); then
    ml_python3 --adhoc_import_modules="${ADHOC_IMPORT_MODULES}" \
      "${MODEL_BASE}/main.py" -- \
      "${args[@]}"
  else
    blaze run -c opt --config=dmtf_cuda \
      "//${MODEL_BASE}:main" -- \
      "${args[@]}"
  fi

  LOG INFO "Stored model output in: ${FLAGS_workdir}"
}
gbash::main "$@"
