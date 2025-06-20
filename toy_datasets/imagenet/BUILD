load("//learning/brain/frameworks/blaze:py3.bzl", "py3_binary", "py3_library", "py3_test")
load("//learning/deepmind/python/fragmented_python:build_defs.bzl", "fragmented_py_binary_mpms")

package(default_visibility = ["//visibility:private"])

exports_files(["launch.py"])

FRAGMENTS = [
    "//learning/brain",
    "//experimental/largesensormodels/toy_datasets/imagenet",  # CHANGEME
    "//third_party/py",
    "//third_party/py/flax",
    "//third_party/py/jax",
]

py3_library(
    name = "input_pipeline",
    srcs = ["input_pipeline.py"],
    deps = [
        "//third_party/py/clu:preprocess_spec",
        "//third_party/py/grain:tensorflow",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_datasets",
    ],
)

py3_test(
    name = "input_pipeline_test",
    srcs = ["input_pipeline_test.py"],
    data = ["//third_party/py/tensorflow_datasets/testing/metadata:imagenet2012"],
    pytype = False,  # Change this to True after checking that the unit test passes.
    deps = [
        ":input_pipeline",
        "//third_party/py/tensorflow",
    ],
)

py3_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":input_pipeline",
        ":resnet_v1",
        "//third_party/py/absl/logging",
        "//third_party/py/clu:metrics",
        "//third_party/py/clu:parameter_overview",
        "//third_party/py/clu:periodic_actions",
        "//third_party/py/clu/metric_writers",
        "//third_party/py/clu/platform",
        "//third_party/py/etils/epath",
        "//third_party/py/flax:core",
        "//third_party/py/grain:tensorflow",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/optax",
        "//third_party/py/orbax/checkpoint",
    ],
)

# Run test with "--config=cuda --compilation_mode=opt".
py3_test(
    name = "train_test",
    srcs = ["train_test.py"],
    data = ["//third_party/py/tensorflow_datasets/testing/metadata:imagenet2012"],
    pytype = False,  # Change this to True after checking that the unit test passes.
    run_on_cpu = False,  # 2-3 slower than GPU or TPU.
    run_on_gpu = True,
    run_on_jf = True,
    deps = [
        ":train",
        "//experimental/largesensormodels/toy_datasets/imagenet/configs",
        "//learning/brain/research/jax:gpu_support",  # build_cleaner: keep
        "//learning/brain/research/jax:tpu_support",  # build_cleaner: keep
        "//third_party/py/absl/logging",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/etils/epath",
        "//third_party/py/grain:tensorflow",
        "//third_party/py/jax",
        "//third_party/py/tensorflow",
    ],
)

py3_binary(
    name = "main",
    srcs = [
        "main.py",
    ],
    deps = [
        ":train",
        "//file/liball",  # build_cleaner: keep
        "//experimental/largesensormodels/toy_datasets/imagenet/configs",  # build_cleaner: keep
        "//learning/brain/research/jax:gpu_support",  # build_cleaner: keep
        "//learning/brain/research/jax:tpu_support",  # build_cleaner: keep
        "//perftools/accelerators/xprof/xprofilez/cpu:python_tracer",  # build_cleaner: keep
        "//perftools/gputools/profiler:xprofilez_with_server",  # build_cleaner: keep
        "//learning/pathways/jax:pathways",
        # gRPC clients and servers often require the Envelope to work correctly.
        # To be able to run the program locally against UPTC, this dependency is required.
        "//tech/env:envelope_loader",  # build_cleaner: keep
        "//third_party/py/clu/platform",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/ml_collections/config_flags",
        "//third_party/py/tensorflow",
    ],
)

# Build rules for go/fragmented-python. This helps to speed up repeated builds
# for XManager experiment launches.
fragmented_py_binary_mpms(
    name = "main_fragmented_mpms",
    binary = ":main",
    fragments = FRAGMENTS,
    package_name_prefix = "ephemeral/experimental/largesensormodels/toy_datasets/imagenet",  # CHANGEME
)

py3_library(
    name = "resnet_v1",
    srcs = [
        "resnet_v1.py",
    ],
    deps = [
        "//third_party/py/flax",
        "//third_party/py/jax",
    ],
)

py3_test(
    name = "resnet_v1_test",
    srcs = ["resnet_v1_test.py"],
    pytype = False,  # Change this to True after checking that the unit test passes.
    deps = [
        ":resnet_v1",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/clu:parameter_overview",
        "//third_party/py/jax",
        "//third_party/py/tensorflow",
    ],
)
