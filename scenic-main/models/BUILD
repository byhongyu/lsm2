load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

pytype_strict_library(
    name = "lsm_vit",
    srcs = ["lsm_vit.py"],
    deps = [
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:model_constants",
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:model_utils",
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:patcher",
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:patcher_config",
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:vit",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/scenic/model_lib/base_models:base_model",
        "//third_party/py/scenic/model_lib/base_models:model_utils",
        "//third_party/py/scenic/model_lib/layers:nn_layers",
        "//third_party/py/scenic/projects/multimask/models:model_utils",
    ],
)

pytype_strict_library(
    name = "lsm_resnet",
    srcs = ["lsm_resnet.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/scenic/common_lib:debug_utils",
        "//third_party/py/scenic/model_lib/base_models:classification_model",
        "//third_party/py/scenic/model_lib/layers",
        "//third_party/py/scenic/projects/baselines:resnet",
    ],
)
