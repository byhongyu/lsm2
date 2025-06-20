"""Patchifying, necessary for "per-modality" kernel"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

from google3.experimental.largesensormodels.scenic.utils.base_config import Base_Config


# TODO(xumax, girishvn) enforce strict type hints in the config
class Patcher_Config(Base_Config):
  """Configuration class for the Patcher module.

  Attributes:
      hidden_size (int): Number of output channels for the convolution layer.
      kernel_size (Union[int, tuple[int], tuple[int]]): Size of the convolution
        kernel. For "2d" mode, it should be a tuple of two integers. For "1d"
        mode, it can be a single integer or a tuple of length one.
      stride (Optional[Union[int, tuple[int], tuple[int, int]]]): Stride size
        for the convolution. If not provided, it defaults to `kernel_size`.
      groups (int): Number of input channels to group for grouped convolution.
        Defaults to 1 (no grouping).
      mode (Literal["2d", "1d"]): Type of convolution mode. "2d" for
        two-dimensional convolution and "1d" for one-dimensional convolution.
        Defaults to "2d".
  """

  def __init__(
      self,
      hidden_size: int,
      kernel_size: Union[int, tuple[int], tuple[int, int]],
      stride: Optional[Union[int, tuple[int], tuple[int, int]]] = None,
      groups: int = 1,
      mode: Literal["2d", "1d"] = "2d",
  ):
    super().__init__()
    self.hidden_size = hidden_size

    self.kernel_size = kernel_size

    if stride is None:
      self.stride = kernel_size
    else:
      self.stride = stride

    self.groups = groups
    self.mode = mode

    if mode == "2d":
      assert isinstance(kernel_size, tuple)
      assert isinstance(self.stride, tuple)
      self.patchsize = kernel_size

    elif mode == "1d":
      assert isinstance(kernel_size, int) or len(kernel_size) == 1
      assert isinstance(self.stride, int) or len(self.stride) == 1

      if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
      self.patchsize = (kernel_size, 1)
