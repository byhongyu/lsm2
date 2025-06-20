"""
THIS BUILD FILE IS UNUSED WHEN CALLING CONFIGS. 

However, it is needed to stop `> Dependences` command from autogenerating the build files

scenic will automatically parse all dependencies throughout the project in a graph style fashion 
from main.py and add them to be visible in the config call. In that way, in order to have a file 
be visible in configs, it must be called in a "real" function that can be used at some point.

Please add imports into utils/dummy.py to make them accessibly in the config file
"""

load("//learning/deepmind/python/config_flags:config_flags_lib.bzl", "config_flags_strict_lib")

# Creates a test case for each config file that it can be parsed.
config_flags_strict_lib(
    name = "config_lib",
    srcs = glob([
        "*.py",
        "*/*.py",
    ]),
    validate_as_architect_plan = False,
    visibility = ["//visibility:private"],
    deps = [
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:model_constants",
        "//experimental/largesensormodels/scenic/models/lsm_vit_utils:patcher_config",
        "//experimental/largesensormodels/scenic/trainers/masking:masker_config",
        "//experimental/largesensormodels/scenic/utils:config_constants",
        "//experimental/largesensormodels/scenic/utils:linear_probe_config",
        "//experimental/largesensormodels/scenic/utils:predefined_configs",
        "//third_party/py/ml_collections",
        "//third_party/py/scenic/projects/baselines/configs:config_lib",
    ],
)
