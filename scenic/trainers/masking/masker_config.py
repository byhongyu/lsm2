"""Masker class and config"""

# TODO(xumax, girishvn) move mask construction from lsm_vit_utils/model_utils into here


from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from google3.experimental.largesensormodels.scenic.utils.base_config import Base_Config


class MaskStrategy_Config(Base_Config):
  """Configuration class for defining a masking strategy.

  Attributes:
      strategy (str): Type of masking strategy to apply. Options include:  -
        "random": Randomly masks elements. - "forecast": Masks elements based on
        forecasting context. - "imputation": Masks elements for imputation
        tasks. - "bar": Applies a structured bar-shaped mask on either the
        "time" or "modality" dimension. - "randompartialbar": Partially applies
        a bar mask on some points along the dimension and with randomness on
        other points in the dimension. Defaults to "random".
      mask_probability (float): Probability of masking each element. Defaults to
        0.8.
      weight (int): Weight of this strategy when choosing among multiple
        strategies. Higher values increase the likelihood of selection. Defaults
        to 1.
      mask_dim (str): Dimension to apply the mask along when using any strategy
        except for "random". Options are: - "time": Masks along the time axis. -
        "modality": Masks along the modality axis. Defaults to "time".
      mask_dim_prob (float, optional): Probability of masking the primary
        dimension when using the "randompartialbar" strategy. Defaults to None.
      mask_offdim_prob (float, optional): Probability of masking the secondary
        dimension when using the "randompartialbar" strategy. Defaults to None.
      mask_dim_forecasting: Optional[bool] Forces the contiguous bar to be at
        the end of the sequence, rather than being randomly applied
      inherited_depend (bool): Whether new artifical masks should depend on
        inherited masks, which should be only turned on for "random" strategies.
        Specifically, we identify the number of mask tokens present in the
        inherited mask, and then add n more mask tokens until we hit the total
        mask probability. When False, requires strictmaskperc to handle varying
        mask amounts.
          Default: True.
  """

  def __init__(
      self,
      strategy: Literal[
          "random", "forecast", "imputation", "bar", "randompartialbar"
      ] = "random",
      mask_probability: float = 0.8,
      weight: int = 1,
      mask_dim: Literal["time", "modality"] = "time",
      mask_dim_contiguous: bool = False,
      mask_dim_forecasting: Optional[bool] = False,
      mask_dim_prob: Optional[
          float
      ] = None,  # only used with strategy="partialbar"
      mask_offdim_prob: Optional[
          float
      ] = None,  # only used with strategy="partialbar"
      inherited_depend: Optional[bool] = None,
  ):
    super().__init__()
    self.strategy = strategy
    self.mask_probability = mask_probability
    self.weight = weight
    self.mask_dim = mask_dim
    self.mask_dim_contiguous = mask_dim_contiguous
    self.mask_dim_forecasting = mask_dim_forecasting
    self.mask_dim_prob = mask_dim_prob
    self.mask_offdim_prob = mask_offdim_prob

    if mask_dim in ["time"]:
      self.mask_dim_idx = 0
    elif mask_dim in ["modality"]:
      self.mask_dim_idx = 1

    if mask_dim_forecasting:
      assert self.mask_dim_contiguous

    self.inherited_depend = inherited_depend


class Masker_Config(Base_Config):
  """Configuration class for controlling masking behavior in transformer models.

  This configuration handles both the selection of masking strategies and their
  application patterns, supporting hybrid approaches that combine:
  - Drop-out removal (ViT-style token removal)
  - Attention masking (BERT-style attention suppression)
  - Inherited masks (from previous processing stages)

  Attributes:
      maskstrategy_list (List[MaskStrategy_Config]): List of masking strategies
        to apply. Each strategy defines: - mask_dim_idx: Dimension to mask along
        (0=time, 1=modality) - strategy: 'random' or 'bar' pattern masking -
        mask_probability: Fraction of tokens to mask - weight: Relative
          weighting when mixing strategies
      on_cpu (bool): If True, forces masking operations to run on CPU.
          Default: False (use GPU if available).
      inherited (bool): Whether to respect existing imputation masks from input.
        When True, combines new masks with any pre-existing masks.
          Default: False.
      mixstrategy (str): Strategy for combining multiple masking approaches: -
        "within_instance": Splits instance along specified dimension (e.g.,
        first half uses Strategy A, second half uses Strategy B) -
        "between_instance": Applies one strategy per instance in batch
          Default: "between_instance"
      strictmaskperc (Optional[float]): Enforces exact masking percentage when
        set, which is the LOWERBOUND of missingness possible. Divides masked
        tokens into two groups: - Attention-masked tokens (variable count,
        processed but ignored in encoder) - Drop-out removed tokens (fixed
        count, excluded from encoder) When None, uses pure attention masking.
        Default: None.
      maskstrategy_weights (np.ndarray): Normalized weights derived from
        strategy weights, used for probabilistic strategy selection.
  """

  def __init__(
      self,
      maskstrategy_list: List[MaskStrategy_Config],
      on_cpu: bool = False,
      inherited: bool = False,
      mixstrategy: Literal[
          "between_instance", "within_instance"
      ] = "between_instance",
      strictmaskperc: Optional[float] = None,
  ):
    super().__init__()
    self.maskstrategy_list = maskstrategy_list
    self.on_cpu = on_cpu
    self.inherited = inherited
    self.mixstrategy = mixstrategy
    self.strictmaskperc = strictmaskperc

    self.maskstrategy_weights = np.array(
        [strat.weight for strat in maskstrategy_list]
    )

    self.run_checks()

  def run_checks(self):
    # Enforce strictmaskperc when using independent inheritance
    for maskstrategy in self.maskstrategy_list:
      if self.inherited:
        assert maskstrategy.inherited_depend is not None, (
            "inherited=True in Masker_Config but inherited_depend is not set"
            " for the maskstrategy"
        )
      if maskstrategy.inherited_depend is not None:
        assert (
            self.inherited != False
        ), "inherited_depend is set but inherited==False in Masker_Config"

        if maskstrategy.inherited_depend is False:
          assert self.strictmaskperc is not None, (
              "strictmaskperc must be set when inherited_depend=False "
              "to handle varying inherited mask amounts"
          )

  def update_maskstrategy_list(self, new_maskstrategy_list):
    """Updates the active masking strategies and recomputes dependent values.

    Args:
        new_maskstrategy_list: Updated list of MaskStrategy_Config objects
    """
    self.maskstrategy_list = new_maskstrategy_list
    self.maskstrategy_weights = np.array(
        [strat.weight for strat in new_maskstrategy_list]
    )

    self.update_strictmaskperc_auto()

    self.run_checks()

  def update_strictmaskperc_auto(self):
    """Automatically sets strictmaskperc to the minimum mask probability

    across all strategies to ensure consistency.
    """
    self.strictmaskperc = np.min(
        np.array([strat.mask_probability for strat in self.maskstrategy_list])
    )
