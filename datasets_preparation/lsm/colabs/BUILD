load("//learning/grp/pynball:build_defs.bzl", "ipynb_deps")

ipynb_deps(
    name = "lsm_v2_data_investigation_simple_deps",
    srcs = ["LSM_v2_data_investigation_simple.ipynb"],
    pytype = False,
    deps = [
        "//medical/waveforms/modelling/lsm/datasets/lsm:sensor_constants",
        "//pyglib:gfile",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/seaborn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_datasets",
    ],
)

ipynb_deps(
    name = "curate_lsm_v2_missing_balanced_dataset_deps",
    srcs = ["curate_lsm_v2_missing_balanced_dataset.ipynb"],
    pytype = False,
    deps = [
        "//pyglib:gfile",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/seaborn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_datasets",
    ],
)

ipynb_deps(
    name = "curate_lsm_v2_missing_breakdown_dataset_deps",
    srcs = ["curate_lsm_v2_missing_breakdown_dataset.ipynb"],
    pytype = False,
    deps = [
        "//pyglib:gfile",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
        "//third_party/py/seaborn",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/tensorflow_datasets",
    ],
)

ipynb_deps(
    name = "generate_user_ids_deps",
    srcs = ["generate_user_ids.ipynb"],
    pytype = False,
    deps = [
        "//medical/waveforms/modelling/lsm/datasets/lsm:sensors",
        "//pyglib:gfile",
        "//third_party/py/matplotlib",
        "//third_party/py/pandas",
    ],
)
