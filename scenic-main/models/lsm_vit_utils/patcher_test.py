"""test for patchifying functions in patch.py"""

from flax.core import freeze, unfreeze
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils.patcher import Patcher_Class
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils.patcher_config import Patcher_Config
from google3.testing.pybase import googletest

BATCH_SIZE = 16
LENGTH = 100
MODALITIES = 32

INPUT_SIMPLE = np.ones((BATCH_SIZE, LENGTH, MODALITIES, 1))
for i in range(MODALITIES):
  INPUT_SIMPLE[:, :, i, :] += i

np.random.seed(10)
INPUT = np.random.rand(BATCH_SIZE, LENGTH, MODALITIES, 1)


class PatcherTest(googletest.TestCase):

  def test_conv2d_basic(self):
    config = Patcher_Config(**{
        "hidden_size": 16,
        "kernel_size": (10, 1),
        "stride": None,
        "groups": 1,
        "mode": "2d",
    })
    conv2d = Patcher_Class(config)
    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv2d.init(rng, INPUT)

    # Forward pass
    out = conv2d.apply(params, INPUT)
    pass

  def test_conv2d_deterministic(self):
    HIDDEN_SIZE = 16
    KERNEL_SIZE = (10, 1)
    config = Patcher_Config(**{
        "hidden_size": HIDDEN_SIZE,
        "kernel_size": KERNEL_SIZE,
        "stride": None,
        "groups": 1,
        "mode": "2d",
    })
    conv2d = Patcher_Class(config)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv2d.init(rng, INPUT_SIMPLE)

    # Manually set the conv kernel weights to fixed values
    params = unfreeze(params)
    params["params"]["embedding"]["kernel"] = jnp.ones((
        10,
        1,
        1,
        16,
    ))  # kernel shape = (kernel_height, kernel_width, input_channels, output_channels)
    params = freeze(params)

    # Forward pass
    # (16, 10, 32, 16)
    out = conv2d.apply(params, INPUT_SIMPLE)

    self.assertEqual(
        out.shape,
        (
            BATCH_SIZE,
            LENGTH // KERNEL_SIZE[0],
            MODALITIES // KERNEL_SIZE[1],
            HIDDEN_SIZE,
        ),
    )

    # Expected output: sum over the receptive field (channels)
    expected_output = np.ones((
        BATCH_SIZE,
        LENGTH // KERNEL_SIZE[0],
        MODALITIES // KERNEL_SIZE[1],
        HIDDEN_SIZE,
    ))
    for i in range(0, LENGTH, KERNEL_SIZE[0]):
      for j in range(0, MODALITIES, KERNEL_SIZE[1]):
        temp = np.sum(
            INPUT_SIMPLE[:, i : i + KERNEL_SIZE[0], j : j + KERNEL_SIZE[1], 0],
            axis=(1, 2),
        )
        expected_output[:, i // KERNEL_SIZE[0], j // KERNEL_SIZE[1], :] = temp

    # Assert output matches expected output
    np.testing.assert_allclose(out, expected_output, rtol=1e-5)

  def test_conv1d_basic(self):
    config = Patcher_Config(**{
        "hidden_size": 16,
        "kernel_size": 10,
        "stride": 10,
        "groups": 1,
        "mode": "1d",
    })
    conv1d = Patcher_Class(config)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv1d.init(rng, INPUT)

    # Forward pass
    out = conv1d.apply(params, INPUT)
    pass

  def test_conv1d_deterministic(self):
    HIDDEN_SIZE = 16
    KERNEL_SIZE = 10
    config = Patcher_Config(**{
        "hidden_size": HIDDEN_SIZE,
        "kernel_size": KERNEL_SIZE,
        "stride": None,
        "groups": 1,
        "mode": "1d",
    })
    conv1d = Patcher_Class(config)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv1d.init(rng, INPUT_SIMPLE)

    # Manually set the conv kernel weights to fixed values
    params = unfreeze(params)
    params["params"]["embedding"]["kernel"] = jnp.ones((
        KERNEL_SIZE,
        MODALITIES,
        HIDDEN_SIZE,
    ))  # kernel shape = (kernel_height, kernel_width, input_channels, output_channels)
    params = freeze(params)

    # Forward pass
    # (16, 32, 1, 16)
    out = conv1d.apply(params, INPUT_SIMPLE)

    # pytest assert or something
    self.assertEqual(
        out.shape,
        (
            BATCH_SIZE,
            LENGTH // KERNEL_SIZE,
            1,
            HIDDEN_SIZE,
        ),
    )

    # Expected output: sum over the receptive field (channels)
    expected_output = np.ones((
        BATCH_SIZE,
        LENGTH // KERNEL_SIZE,
        1,
        HIDDEN_SIZE,
    ))
    for i in range(0, LENGTH, KERNEL_SIZE):
      temp = np.sum(
          INPUT_SIMPLE[:, i : i + KERNEL_SIZE, :, 0],
          axis=(1, 2),
      )
      expected_output[:, i // KERNEL_SIZE, 0, :] = temp

    # Assert output matches expected output
    np.testing.assert_allclose(out, expected_output)

  def test_conv1d_permodality_deterministic(self):

    HIDDEN_SIZE = 16 * MODALITIES
    KERNEL_SIZE = 10
    config = Patcher_Config(**{
        "hidden_size": HIDDEN_SIZE,
        "kernel_size": KERNEL_SIZE,
        "stride": None,
        "groups": MODALITIES,
        "mode": "1d",
    })
    conv1d = Patcher_Class(config)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv1d.init(rng, INPUT_SIMPLE)

    # Manually set the conv kernel weights to fixed values
    params = unfreeze(params)
    temp = np.ones((
        10,
        1,
        512,
    ))  # kernel shape = (kernel_height, kernel_width, input_channels, output_channels)

    # given modality always has a specific kernel of size 16
    for idx, i in enumerate(range(0, HIDDEN_SIZE, 16)):
      temp[:, :, i : i + 16] = idx + 1

    params["params"]["embedding"]["kernel"] = temp
    params = freeze(params)

    # Forward pass
    # (16, 32, 16)
    out = conv1d.apply(params, INPUT_SIMPLE)

    # pytest assert or something
    self.assertEqual(
        out.shape,
        (
            BATCH_SIZE,
            LENGTH // KERNEL_SIZE,
            MODALITIES,
            HIDDEN_SIZE // MODALITIES,
        ),
    )

    # Expected output: sum over the receptive field (channels)
    expected_output = np.ones((
        BATCH_SIZE,
        LENGTH // KERNEL_SIZE,
        MODALITIES,
        HIDDEN_SIZE // MODALITIES,
    ))
    for i in range(0, LENGTH, KERNEL_SIZE):
      for j in range(MODALITIES):
        temp = np.sum(
            INPUT_SIMPLE[:, i : i + KERNEL_SIZE, j, :],
            axis=(1),
        ) * (j + 1)
        expected_output[:, i // KERNEL_SIZE, j, :] = temp

    # Assert output matches expected output
    np.testing.assert_allclose(out, expected_output, rtol=1e-5)

  def test_conv1d_permodality_comparenaiveconv1d(self):
    FEATURES = 16
    HIDDEN_SIZE = FEATURES * MODALITIES
    KERNEL_SIZE = 10
    config = Patcher_Config(**{
        "hidden_size": HIDDEN_SIZE,
        "kernel_size": KERNEL_SIZE,
        "stride": None,
        "groups": MODALITIES,
        "mode": "1d",
    })
    conv1d = Patcher_Class(config)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv1d.init(rng, INPUT)
    # Manually set the conv kernel weights to fixed values
    params = unfreeze(params)
    temp = np.ones((
        10,
        1,
        HIDDEN_SIZE,
    ))  # kernel shape = (kernel_height, kernel_width, input_channels, output_channels)
    # given modality always has a specific kernel of size 16
    for idx, i in enumerate(range(0, HIDDEN_SIZE, FEATURES)):
      temp[:, :, i : i + FEATURES] = idx + 1
    params["params"]["embedding"]["kernel"] = temp
    params = freeze(params)
    out = conv1d.apply(params, INPUT)

    conv1d_naive = nn.Conv(
        features=FEATURES,
        kernel_size=KERNEL_SIZE,
        strides=KERNEL_SIZE,
        feature_group_count=1,
        padding="VALID",
        name="embedding",
    )
    input_naive = INPUT[:, :, :1, 0]
    rng_naive = jax.random.PRNGKey(0)
    params_conv1d_naive = conv1d_naive.init(rng_naive, input_naive)

    out_naive = []
    for i in range(MODALITIES):
      params_conv1d_naive = unfreeze(params_conv1d_naive)
      temp = (
          np.ones((
              10,
              1,
              FEATURES,
          ))
          + i
      )
      params_conv1d_naive["params"]["kernel"] = temp
      params_conv1d_naive = freeze(params_conv1d_naive)
      out_naive.append(
          conv1d_naive.apply(params_conv1d_naive, INPUT[:, :, i : i + 1, 0])
      )
    out_naive = np.stack(out_naive, axis=2)
    np.testing.assert_allclose(out, out_naive, rtol=1e-5)

  # test used to make sure that the reshape for conv1d works to put modality in
  # the correct place
  def test_conv1d_permodality_biginput(self):
    biginput = INPUT
    BIGMODALITYINPUT = 1
    biginput[:, :, BIGMODALITYINPUT, :] = 100000000
    FEATURES = 16
    HIDDEN_SIZE = FEATURES * MODALITIES
    KERNEL_SIZE = 10
    config = Patcher_Config(**{
        "hidden_size": HIDDEN_SIZE,
        "kernel_size": KERNEL_SIZE,
        "stride": None,
        "groups": MODALITIES,
        "mode": "1d",
    })
    conv1d = Patcher_Class(config)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv1d.init(rng, INPUT)
    # Manually set the conv kernel weights to fixed values
    out = conv1d.apply(params, INPUT)
    out = jnp.abs(out)

    flat = out.ravel()
    top_indices = jnp.argsort(-flat)[
        : BATCH_SIZE * LENGTH // KERNEL_SIZE * FEATURES
    ]
    top_indices = jnp.unravel_index(top_indices, out.shape)
    assert np.allclose(np.unique(top_indices[2]), BIGMODALITYINPUT)

  def test_conv1d_permodality_sameoutpermodality(self):
    HIDDEN_SIZE = 16 * MODALITIES
    KERNEL_SIZE = 10
    config = Patcher_Config(**{
        "hidden_size": HIDDEN_SIZE,
        "kernel_size": KERNEL_SIZE,
        "stride": None,
        "groups": MODALITIES,
        "mode": "1d",
    })
    conv1d = Patcher_Class(config)

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    params = conv1d.init(rng, INPUT_SIMPLE)

    # Forward pass
    # (16, 32, 16)
    out = conv1d.apply(params, INPUT_SIMPLE)

    self.assertEqual(
        out.shape,
        (
            BATCH_SIZE,
            LENGTH // KERNEL_SIZE,
            MODALITIES,
            HIDDEN_SIZE // MODALITIES,
        ),
    )

    # Expected output: sum over the receptive field (channels)
    expected_output = np.ones((
        BATCH_SIZE,
        LENGTH // KERNEL_SIZE,
        MODALITIES,
        HIDDEN_SIZE // MODALITIES,
    ))

    # Assert output matches expected output
    for i in range(MODALITIES):
      for j in range(HIDDEN_SIZE // MODALITIES):
        self.assertTrue((out[:, :, i, j] == out[0, 0, i, j]).all())


if __name__ == "__main__":
  googletest.main()
