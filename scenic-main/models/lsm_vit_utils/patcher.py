"""Patchifying, necessary for "per-modality" kernel"""

import functools
from typing import Any, Dict, Literal, Optional, Tuple, Union

import flax.linen as nn
import ml_collections
# # importing this file from the configs breaks xm job launching due to the
# # "import jax.numpy as jnp" so we leave it commented out
# import jax.numpy as jnp
import numpy as np

from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils.patcher_config import Patcher_Config


# TODO(xumax): Modify this conv layer to support partial convolutions
# https://github.com/taki0112/partial_conv-Tensorflow
# https://github.com/MathiasGruber/PConv-Keras
class Patcher_Class(nn.Module):
  """Patcher module for patching multimodal time-series data using 1D or 2D convolutions.

  Attributes:
      config (Patcher_Config): Configuration object that defines the convolution
        parameters, including kernel size, stride, grouping, and mode.

  Methods:
      __call__(x): Performs the forward pass using the specified convolution
        mode, "1d" or "2d".

  Args:
      x (np.ndarray): Input tensor of shape [batch_size, time, modalities, 1]. -
        batch_size: Number of samples in the batch. - time: Temporal dimension.
          - modalities: Number of input modalities. - 1: Dummy channel dimension
          (analogous to RGB channels in images).

  Returns:
      np.ndarray: Output tensor after applying the convolution and reshaping
          (if in 1D mode).

  Raises:
      ValueError: If the mode specified in the configuration is not "1d" or
      "2d".
  """

  config: Patcher_Config

  @nn.compact
  def __call__(self, x):
    """Forward pass of Patcher for ViT.

    Args:
        x: Input multimodal time-series tensor. Shape [batch_size, time,
          modalities, 1], where 1 is the dummy channel dimension (where rgb
          would be for images).
    """
    if self.config["mode"] == "2d":

      x = nn.Conv(
          features=self.config["hidden_size"],
          kernel_size=self.config["kernel_size"],
          strides=self.config["stride"],
          feature_group_count=self.config["groups"],
          padding="VALID",
          name="embedding",
      )(x)
      return x

    elif self.config.mode == "1d":

      x = x.squeeze(axis=-1)  # conv1d expects B, T, C
      x = nn.Conv(
          features=self.config["hidden_size"],
          kernel_size=self.config["kernel_size"],
          strides=self.config["stride"],
          feature_group_count=self.config["groups"],
          padding="VALID",
          name="embedding",
      )(x)

      # in per_modality approach, groups is == number of modalities and
      # hidden_size is original hidden size * groups
      batch, time, groups_and_channels = x.shape
      x = np.reshape(
          x,
          [
              batch,
              time,
              self.config.groups,
              self.config.hidden_size // self.config.groups,
          ],
      )
    else:
      raise ValueError(
          'defined Patcher_Config mode does not use "1d" nor "2d". This should'
          " be caught in the config type hints, but it is not enforced strictly"
          " yet."
      )
    return x
