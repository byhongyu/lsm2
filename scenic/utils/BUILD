load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

pytype_strict_library(
    name = "linear_probe_config",
    srcs = ["linear_probe_config.py"],
    deps = ["//third_party/py/ml_collections"],
)

pytype_strict_library(
    name = "classification_utils",
    srcs = ["classification_utils.py"],
    deps = [
        "//pyglib:gfile",
        "//third_party/py/clu/metric_writers",
        "//third_party/py/jax",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/sklearn",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "config_constants",
    srcs = ["config_constants.py"],
)

pytype_strict_library(
    name = "base_config",
    srcs = ["base_config.py"],
    deps = ["//third_party/py/ml_collections"],
)

pytype_strict_library(
    name = "dummy",
    srcs = ["dummy.py"],
    deps = [
        ":config_constants",
        ":predefined_configs",
        "//experimental/largesensormodels/scenic/trainers/masking:masker_config",
    ],
)

pytype_strict_library(
    name = "predefined_configs",
    srcs = ["predefined_configs.py"],
    deps = [
        ":base_config",
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:model_constants",
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:patcher_config",
        "//experimental/largesensormodels/scenic/trainers/masking:masker_config",
    ],
)
