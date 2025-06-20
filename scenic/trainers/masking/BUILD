load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

pytype_strict_library(
    name = "masker",
    srcs = ["masker.py"],
    deps = [
        ":masker_config",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/scenic/dataset_lib:dataset_utils",
        "//third_party/py/tensorflow:tensorflow_core",
    ],
)

pytype_strict_library(
    name = "masker_config",
    srcs = ["masker_config.py"],
    deps = [
        "//experimental/largesensormodels/scenic/utils:base_config",
        "//third_party/py/numpy",
    ],
)
