# Brain Template : JAX ImageNet

authors: mrit@, andstein@

This file only covers this specific template. Please refer to
go/brain-templates-doc for all other information.

**Fork** this template by executing:

```bash
blaze run -c opt //learning/brain/frameworks/templates:fork -- \
    --template=jax/imagenet \
    --destination=experimental/users/$USER/imagenet
```

## Current Status

Supported environments:

Configuration            | local | Borg | Colab | GCP
------------------------ | ----- | ---- | ----- | ---
Single worker single GPU | ✅     | ✅    | ✅     | ✅
Single worker multi GPU  | ✅     | ✅    | ✅     | ✅
TPU Donut (single host)  | ✅     | ✅    | ✅     | ❌
TPU Pod (multiple hosts) | ❌     | ✅    | ❌     | ❌

Observed runtimes and accuracies:

Accelerator | Slice | Runtime | Final accuracy | XID
----------- | ----- | ------- | -------------- | --------------
Jellyfish   | 2x2   | 11h     | 0.76333        |
Jellyfish   | 4x4   | 4h      | 0.76989        |
Dragonfish  | 4x4   | 2h33m   | 0.76820        | xid/56075923/1
Dragonfish  | 8x8   | 35m     | 0.76296        | xid/17012436/1
Pufferfish  | 4x4x4 | 38m     | 0.76344        | xid/43075402

## Example Runs

**Launching** the forked template in cell `tp` on a Dragonfish 8x8 slice:

```bash
xmanager launch experimental/largesensormodels/toy_datasets/imagenet/launch.py -- \
    --xm_resource_alloc=group:cml/cml-shared-ml-user \
    --platform=df=8x8 --config=default.py
```

From above run:

-   xid/49085650
    https://cnsviewer2.corp.google.com//cns/je-d/home/andstein/xm/49085650/1
-   final accuracy : 0.76332, runtime : 23 minutes
    https://flatboard.corp.google.com/plot/90ABDlvbWCo
-   92.4% duty cycle
    https://xprof.corp.google.com/overview_page/andstein-9283854156374733157

Regression tests:

go/bt-regression

https://colab.corp.google.com/drive/1v5nnzJ8512Yp78Oz__HJtSHTP8OohWF7

```
%xolab
X.search('jax.imagenet.df_4x4', tags=('regression-test',))
```
