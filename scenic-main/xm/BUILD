load("//learning/brain/frameworks/blaze:py3.bzl", "py3_library")

package(
    default_applicable_licenses = ["//third_party/py/scenic:license"],
    default_visibility = ["//third_party/py/scenic:scenic_friends"],
)

licenses(["notice"])

py3_library(
    name = "xm_utils",
    srcs = ["xm_utils.py"],
    deps = [
        "//learning/deepmind/analysis/flatboard",
        "//learning/deepmind/xmanager2/client:xmanager_api",
        "//learning/deepmind/xmanager2/client/google",
        "//pyglib:gfile",
        "//third_party/py/absl/logging",
        "//third_party/py/ml_collections",
    ],
)

py3_library(
    name = "launch_utils",
    srcs = ["launch_utils.py"],
    deps = [
        "//learning/deepmind/analysis/flatboard",
        "//learning/deepmind/python/adhoc_import:binary_import",
        "//learning/deepmind/xmanager:hyper",
        "//learning/vizier/service/client/pyvizier",
        "//third_party/py/ml_collections",
        "//third_party/py/xmanager/xm:__init__",
        "//third_party/py/xmanager/xm_abc:__init__",
    ],
)

# Build target for XManager regression tests (see learning/brain/testing/xmanager/BUILD):
py3_library(
    name = "integration_test_deps",
    deps = [
        ":launch_utils",
        ":xm_utils",
        "//third_party/py/scenic",
        "//third_party/py/scenic/projects/baselines/configs:config_lib",
        "//third_party/py/scenic/projects/vit_plus_v2:main_lib",
        "//third_party/py/scenic/projects/vit_plus_v2/configs",
    ],
)

# Make sure that the launcher is "visible" for the regression tests system:
exports_files([
    "launch_xm.py",
])
