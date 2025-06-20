load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

pytype_strict_library(
    name = "lsm_v2_pretraining_dataset",
    srcs = ["lsm_v2_pretraining_dataset.py"],
    deps = [
        ":dataset_constants",
        ":dataset_utils",
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/scenic/dataset_lib:dataset_utils",
        "//third_party/py/tensorflow:tensorflow_core",
        "//third_party/py/tensorflow_datasets",
    ],
)

pytype_strict_library(name = "lsm_v2_generative_eval_dataset")

pytype_strict_library(
    name = "lsm_v2_metabolic_health_dataset",
    srcs = ["lsm_v2_metabolic_health_dataset.py"],
    deps = [
        ":dataset_constants",
        ":dataset_utils",
        ":lsm_v2_pretraining_dataset",
        "//pyglib:gfile",
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/scenic/dataset_lib:dataset_utils",
        "//third_party/py/tensorflow:tensorflow_core",
    ],
)

pytype_strict_library(
    name = "get_dataset",
    srcs = ["get_dataset.py"],
    deps = [
        ":lsm_v2_metabolic_health_dataset",
        ":lsm_v2_pretraining_dataset",
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/scenic/dataset_lib:dataset_utils",
    ],
)

pytype_strict_library(
    name = "dataset_constants",
    srcs = ["dataset_constants.py"],
)

pytype_strict_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.py"],
    deps = [
        "//experimental/largesensormodels/scenic/trainers/masking:masker",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_core",
    ],
)
