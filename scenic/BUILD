load("//third_party/py/scenic:build_defs.bzl", "scenic_main")

scenic_main(
    name = "main",
    srcs = ["main.py"],
    deps = [
        "//experimental/largesensormodels/scenic/datasets:get_dataset",
        "//experimental/largesensormodels/scenic/models:lsm_resnet",
        "//experimental/largesensormodels/scenic/models:lsm_vit",
        "//experimental/largesensormodels/scenic/trainers:lsm_embedding_dump",
        "//experimental/largesensormodels/scenic/trainers:lsm_generative_task_standalone_eval",
        "//experimental/largesensormodels/scenic/trainers:lsm_linear_probe_standalone_eval",
        "//experimental/largesensormodels/scenic/trainers:lsm_linear_probe_standalone_trainer",
        "//experimental/largesensormodels/scenic/trainers:lsm_mae_trainer",
        "//experimental/largesensormodels/scenic/trainers:lsm_supervised_standalone_eval",
        "//experimental/largesensormodels/scenic/trainers:lsm_supervised_trainer",
        "//experimental/largesensormodels/scenic/utils:dummy",
        "//experimental/largesensormodels/scenic/utils:linear_probe_config",
        "//third_party/py/absl/flags",
        "//third_party/py/clu/metric_writers",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/scenic:app",
        "//third_party/py/scenic/projects/baselines:simple_cnn",
        "//third_party/py/scenic/projects/baselines:vit",
        "//third_party/py/tensorflow:tensorflow_core",
    ],
)
