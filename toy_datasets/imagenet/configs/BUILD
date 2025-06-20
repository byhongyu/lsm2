load("//learning/brain/frameworks/blaze:py3.bzl", "py3_library", "py3_test")
load("//learning/deepmind/python/config_flags:config_flags_lib.bzl", "config_flags_strict_lib")

package(default_visibility = [
    "//experimental/largesensormodels/toy_datasets/imagenet:__subpackages__",
    "//learning/brain/testing/xmanager:__pkg__",
])

# Creates a test case for each config file that it can be parsed.
config_flags_strict_lib(
    name = "configs",
    srcs = glob(
        ["*.py"],
        exclude = [
            "common.py",
            "*_test.py",
        ],
    ),
    validate_as_architect_plan = False,
    deps = [
        ":common",
        "//third_party/py/ml_collections",
    ],
)

py3_library(
    name = "common",
    srcs = [
        "common.py",
    ],
    deps = [
        "//third_party/py/ml_collections",
    ],
)

py3_test(
    name = "common_test",
    srcs = ["common_test.py"],
    pytype = False,  # Change this to True after checking that the unit test passes.
    deps = [
        ":common",
        "//testing/pybase",
        "//third_party/py/absl/testing:parameterized",
    ],
)
