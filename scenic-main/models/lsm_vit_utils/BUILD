load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

pytype_strict_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/scenic/dataset_lib:dataset_utils",
    ],
)

py_strict_test(
    name = "patcher_test",
    srcs = ["patcher_test.py"],
    deps = [
        ":patcher",
        ":patcher_config",
        "//testing/pybase",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "patcher",
    srcs = ["patcher.py"],
    deps = [
        ":patcher_config",
        "//experimental/largesensormodels/scenic/utils:base_config",
        "//third_party/py/flax:core",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "patcher_config",
    srcs = ["patcher_config.py"],
    deps = ["//experimental/largesensormodels/scenic/utils:base_config"],
)

pytype_strict_library(
    name = "vit",
    srcs = ["vit.py"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/scenic/model_lib/layers:attention_layers",
        "//third_party/py/scenic/model_lib/layers:nn_layers",
        "//third_party/py/scipy",
    ],
)

pytype_strict_library(
    name = "model_constants",
    srcs = ["model_constants.py"],
)
