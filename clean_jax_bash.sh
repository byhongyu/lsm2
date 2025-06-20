#!/bin/bash
# Script for deleting jax caches.
#
# Usage:
#   bash experimental/largesensormodels/clean_jax_caches.sh
#
# This script will delete all jax caches in the current workspace.
#
# Note:
#   This script will not delete caches in the current workspace if the current
#   workspace is not a citc workspace.
#
#   This script will not delete caches in the current workspace if the current
#   workspace is not a citc workspace.
#
#   This script will not delete caches in the current workspace if the current
#   workspace is not

source gbash.sh || exit
source module lib/colors.sh
DEFINE_string --alias=l ldap xliucs "Your Google LDAP."

CELLS=("ed" "cg" "eq" "ge")
for CELL in "${CELLS[@]}"; do
  echo "Deleting jax caches in cell ${CELL}."
  fileutil rm -R -f --parallelism=20 /cns/${CELL}-d/home/${FLAGS_ldap}/jax
done

