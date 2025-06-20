# lsm dataset

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//learning/brain/frameworks/blaze:py3.bzl", "py3_library")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/py/tensorflow_datasets/google:build_defs.bzl", "download_and_prepare_binary")
load("//tools/build_defs/borgcfg:borgcfg_library.bzl", "borgcfg_library")

package(
    default_visibility = [":internal"],
)

package_group(
    name = "internal",
    packages = [
        "//medical/waveforms/...",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "experiment_constants",
    srcs = ["experiment_constants.py"],
)

py3_library(
    name = "lsm",
    srcs = ["lsm.py"],
    deps = [
        ":constants",
        ":experiment_constants",
        ":sensors",
        "//pyglib:gfile",
        "//third_party/py/apache_beam",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_datasets",
        "//third_party/py/typing_extensions",
    ],
)

py3_library(
    name = "datasets",
    srcs = ["__init__.py"],
    visibility = [
        ":internal",
    ],
    deps = [
        ":lsm",
    ],
)

download_and_prepare_binary(
    name = "download_and_prepare",
    # Imports executed to register the datasets.
    dataset_imports = [
        "google3.medical.waveforms.modelling.lsm.datasets.lsm",
    ],
    # Extra deps required to generate specific datasets.
    deps = [
        ":datasets",
        "//file/scratch",
    ],
)

genmpm(
    name = "download_and_prepare_mpm",
    package_name = "medical/waveforms/modelling/lsm/datasets/lsm/download_and_prepare",
    srcs = [
        ":download_and_prepare.par",
    ],
)

borgcfg_library(
    name = "download_and_prepare_borg",
    srcs = ["download_and_prepare.borg"],
    deps = ["//production/borg/templates:mpm_lib"],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_library(
    name = "sensor_constants",
    srcs = ["sensor_constants.py"],
)

py_library(
    name = "sensors",
    srcs = ["sensors.py"],
    visibility = [
        ":internal",
    ],
    deps = [
        ":constants",
        ":experiment_constants",
        ":sensor_constants",
        ":sensors_caption",
        "//fitbit/research/sensing/common/colab:metadata_database_helpers",
        "//fitbit/research/sensing/common/infra/transforms:data_loading",
        "//fitbit/research/sensing/common/infra/utils:common_data_helper",
        "//fitbit/research/sensing/common/infra/utils:constants",
        "//fitbit/research/sensing/common/infra/utils:data_intermediates",
        "//fitbit/research/sensing/common/proto:data_key_py_pb2",
        "//fitbit/research/sensing/kereru/utils:data_loader",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_strict_test(
    name = "sensors_test",
    srcs = ["sensors_test.py"],
    deps = [
        ":sensors",
        "//testing/pybase",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

pytype_strict_library(
    name = "sensors_caption",
    srcs = ["sensors_caption.py"],
    deps = [
        ":sensor_constants",
        "//third_party/py/numpy",
    ],
)
