"""ViT encoder-decoder models for LSM.

Adapted from google3/third_party/py/scenic/projects/multimask/models/vit_mae.py.
"""

import functools
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.layers import nn_layers
from scenic.projects.multimask.models import model_utils as mm_model_utils

# from scenic.projects.baselines import vit
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_constants  # pylint: disable=unused-import
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import model_utils as lsm_model_utils
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import patcher
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import patcher_config  # pylint: disable=unused-import
from google3.experimental.largesensormodels.scenic.models.lsm_vit_utils import vit


# Mostly copied from ViTMaskedAutoencoder in projects/mfp/vit.py
class ViTMAE(nn.Module):
  """Encoder-decoder Vision Transformer model for masked feature regression.

  The differences to `ViTMaskedModel` from vit_encoder.py are that:
  -- Only non-masked tokens are processed by the encoder
  -- The parallel decoder then processes all tokens

  Attributes:
    config: Experiment config. Putting it here helps future proof additional
      modifications to the config for model tweaks. Contains Patcher_Config obj.
    num_classes: Number of output classes.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_layers: Number of layers.
    num_heads: Number of self-attention heads.
    patches: Configuration of the patches extracted in the stem of the model.
    hidden_size: Size of the hidden state of the output of model's stem.
    representation_size: Size of the representation layer in the model's head.
      if None, we skip the extra projection + tanh activation at the end.
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: Probability of dropping out a layer during training.
    classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
      'token'.
    dtype: JAX data type for activations.
  """

  config: ml_collections.ConfigDict
  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  decoder_config: ml_collections.ConfigDict
  representation_size: Optional[int] = None
  positional_embedding: str = 'sinusoidal_2d'
  positional_embedding_decoder: str = 'sinusoidal_2d'
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'none'
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.patcher = patcher.Patcher_Class(self.config.model.patcher_config)

  @nn.compact
  def __call__(
      self,
      x: jnp.ndarray,
      mask_indices: Optional[jnp.ndarray] = None,
      unmasked_indices: Optional[jnp.ndarray] = None,
      token_mask: Optional[jnp.ndarray] = None,
      attn_mask: Optional[jnp.ndarray] = None,
      *,
      train: bool,
      debug: bool = False,
  ):
    """Forward pass of Vision Transformer.

    Args:
      x: Input multimodal time-series tensor. Shape [batch_size, time,
        modalities].
      mask_indices: None or Int indices of masked positions. Shape [batch_size,
        n_masked]. This annd the next 2 arguments are only used if
        config.mask_oncpu == True, causing masks to be generated on the cpu
        dataloader
      unmasked_indices: None or Int indices of unmasked positions. Shape
        [batch_size, n_tokens - n_masked].
      token_mask: None Mask indicating masked positions. Shape [batch_size,
        n_tokens]. *: Additional arguments (not used).
      attn_mask: None or Int indices of masked positions. Shape of
        [batch_size, 1, num encoded tokens]
      train: Whether the model is in training mode.
      debug: Whether to run in debug mode.

    Returns:
      A tuple containing:
        - Reconstructed multimodal time-series tensor. Shape [batch_size, time,
        modalities, channels].
        - Dictionary of auxiliary outputs.
    """
    batch, time, modalities, _ = x.shape  # pylint: disable=unused-variable
    f_time, f_modalities = self.config.model.patcher_config.patchsize
    assert time % f_time == 0 and modalities % f_modalities == 0, (
        'time and modalities should be divisible by the respective patch sizes,'
        f' instead got {x.shape[1:3]} and {(f_time, f_modalities)}'
    )

    # TODO(xumax, girishvn): in order to save compute, write code to convolve
    # embed non-masked inputs. requires moving the mask first, changing
    # positional embedding, and making sure stride is equal to patch size...
    x = self.patcher(x)
    batch, time, modalities, channels = x.shape
    n_tokens = time * modalities

    # Add positional encodings before removing the masked tokens
    if '1d' in self.positional_embedding:
      newx_all = []
      for i in range(modalities):
        newx = mm_model_utils.add_positional_embeddings(
            x[:, :, i, :], self.positional_embedding,
        )
        newx_all.append(newx[:, :, None, :])  # add modality dim back
      x = jnp.concatenate(newx_all, axis=2)
      assert x.shape == (batch, time, modalities, channels)
      x = jnp.reshape(x, [batch, n_tokens, channels])
    else:
      x = jnp.reshape(x, [batch, n_tokens, channels])
      x = mm_model_utils.add_positional_embeddings(
          x, self.positional_embedding, [batch, time, modalities, channels],
      )

    # Use CPU-bound masking.
    # Ensure masking info exists.
    if self.config.masker_config.on_cpu:
      assert token_mask is not None
      assert mask_indices is not None
      assert unmasked_indices is not None

    # Construct Masks and Remove Masked Patches (accelerator-bound)
    else:
      # Implies that masking info was NOT passed from the cpu dataloader.
      # Masking is done here in the model.
      assert token_mask is None
      assert mask_indices is None
      assert unmasked_indices is None
      # and now we must construct them here...

      if train:
        assert (
            len(self.config.masker_config.maskstrategy_list) == 1
        ), 'on jax masking does not support multiple mask strategies'

        mask_strat_config = self.config.masker_config.maskstrategy_list[0]
        masking_strategy = mask_strat_config.strategy
        mask_dim = mask_strat_config.mask_dim
        mask_probability = mask_strat_config.mask_probability
        mask_dim_prob = mask_strat_config.mask_dim_prob
        mask_offdim_prob = mask_strat_config.mask_offdim_prob

        # TODO(xumax, girishvn) REFACTOR MASK CODE SO IT IS LESS UGLY
        # it is only written in this way to minimally change the code so i make
        # sure i am not breaking anything
        if mask_dim in ['h', 'time']:
          mask_dim = 'h'
          mask_dim_len = time
          mask_offdim_len = modalities
        elif mask_dim in ['w', 'feature', 'sensor', 'modality']:
          mask_dim = 'w'
          mask_dim_len = modalities
          mask_offdim_len = time
        else:
          raise ValueError(f'Unsupported mask_dim: {mask_dim}')

        # Generate mask indices.
        if masking_strategy == 'random':  # Random Mask
          n_masked = int(mask_probability * n_tokens)
          mask_indices, unmasked_indices, token_mask = (
              mm_model_utils.get_mask_indices(
                  batch, n_tokens, n_masked, self.make_rng('dropout')
              )
          )
        elif masking_strategy == 'forecast':  # Forecast
          n_dim_masked = int(mask_probability * mask_dim_len)
          mask_indices, unmasked_indices, token_mask = (
              lsm_model_utils.get_forecast_mask_indices(
                  n_batch=batch,
                  n_h=time,
                  n_w=modalities,
                  n_dim_masked=n_dim_masked,
                  mask_dim=mask_dim,
              )
          )
        elif masking_strategy == 'imputation':  # Imputation
          n_dim_masked = int(mask_probability * mask_dim_len)
          mask_indices, unmasked_indices, token_mask = (
              lsm_model_utils.get_imputation_mask_indices(
                  n_batch=batch,
                  n_h=time,
                  n_w=modalities,
                  n_dim_masked=n_dim_masked,
                  mask_dim=mask_dim,
                  rng=self.make_rng('dropout'),
              )
          )
        elif masking_strategy == 'bar':  # Structured Bar
          n_dim_masked = int(mask_probability * mask_dim_len)
          mask_indices, unmasked_indices, token_mask = (
              lsm_model_utils.get_random_bar_mask_indices(
                  n_batch=batch,
                  n_h=time,
                  n_w=modalities,
                  n_dim_masked=n_dim_masked,
                  mask_dim=mask_dim,
                  rng=self.make_rng('dropout'),
              )
          )
        elif masking_strategy == 'partialbar':  # Structured Bar
          n_dim_masked = int(mask_dim_prob * mask_dim_len)
          n_offdim_masked = int(mask_offdim_prob * mask_offdim_len)
          mask_indices, unmasked_indices, token_mask = (
              lsm_model_utils.get_random_partial_bar_mask_indices(
                  n_batch=batch,
                  n_h=time,
                  n_w=modalities,
                  n_dim_masked=n_dim_masked,
                  n_offdim_masked=n_offdim_masked,
                  mask_dim=mask_dim,
                  rng=self.make_rng('dropout'),
              )
          )
        else:
          raise ValueError(f'Unsupported masking strategy: {masking_strategy}')

        # Process only unmasked tokens with the encoder.
        assert len(unmasked_indices.shape) == 2
        assert len(mask_indices.shape) == 2

        assert token_mask.size == mask_indices.size + unmasked_indices.size
        assert unmasked_indices.size > 0
        assert mask_indices.size > 0

        # without this, it throws an error on server but not locally
        unmasked_indices = unmasked_indices.astype(int)
        mask_indices = mask_indices.astype(int)
      else:
        # train == False when using the model in embedding mode
        token_mask = jnp.zeros((batch, n_tokens))
        mask_indices = jnp.array([])
        unmasked_indices = jnp.repeat(
            jnp.arange(n_tokens)[None, :], repeats=batch, axis=0
        )

    batch_indices = jnp.arange(batch).reshape(batch, 1).astype(int)
    x = x[batch_indices, unmasked_indices]

    # Save out token mask.
    aux = {'token_mask': token_mask}

    # If we want to add a class token, add it here.
    # Note that in MAE, positional encodings are not added to the CLS token.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, channels), x.dtype)
      cls = jnp.tile(cls, [batch, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
      # aux['cls_token'] = cls  # TODO GIRISH REMOVED FOR TESTING

    # Encoder.
    if attn_mask is not None:
      # prior shape is [batch_size, 1, encoder_len]
      # we reshape to [batch_size, 1, 1, encoder_len] to match attn mask's
      # expected shape of [batch_size, num_heads, query_len, key_len] and the
      # 1s are then broadcasted in the attention mechanism
      attn_mask_encoder = attn_mask[:, None, :, :]
    else:
      attn_mask_encoder = None

    x = vit.Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        positional_embedding='none',  # Has already been added.
        name='Transformer',
    )(x, attn_mask=attn_mask_encoder, train=train)
    aux['pre_logits'] = x

    # If not training, skip decoding
    if not train:
      return x, aux

    # Process entire sequence with the decoder.
    mask_token = self.param(
        'mask_token',
        nn.initializers.zeros,
        (1, 1, self.decoder_config.hidden_size),
    )

    # Decoder projection.
    x = nn.Dense(
        self.decoder_config.hidden_size,
        kernel_init=nn.initializers.xavier_uniform(),
        name='decoder_projection',
    )(x)
    aux['intermediate_projection'] = x
    if self.classifier == 'token':
      x = x[:, 1:, :]

    # ** Unshuffling Tokens to re-add mask tokens in **
    # Please refer to mask_example in:
    # experimental/largesensormodels/scenic/datasets/dataset_utils.py
    # for additional details.

    # Our total mask looks like this
    # [0 1 1 1 0] (meaning indices 1, 2, 3 are masked)
    # Our input_signal sequence looks like this
    # [a b c d e] (meaning there are a total of 5 tokens)
    # Input un/masked indices are:
    # unmasked_indices=[0, 1, 2, 4] mask_indices = [3] (mean index 3 is masked,
    # meaning only a, b, c, and e are passed to the encoder)
    # attn_mask looks like this
    # [1 0 0 1] (meaning b, c are masked in the encoder).

    # **** x is [Ea Eb Ec Ee] ****
    # where En is the encoder output of n

    # **** x_all is [Ea Eb Ec _ Ee] ****
    x_all = jnp.zeros((batch, n_tokens, self.decoder_config.hidden_size))
    x_all = x_all.at[batch_indices, unmasked_indices].set(x)

    # **** x_all is [Ea M M _ Ee] ****
    # where M is the mask token
    if attn_mask is not None:
      n_enc_tokens = unmasked_indices.shape[-1]  # num tokens passed to encoder # pytype: disable=attribute-error
      assert attn_mask.shape[-1] == n_enc_tokens  # pytype: disable=attribute-error

      # attn_mask is of shape [batch_size, 1, encoder_len].
      # We flatten to get a mask of shape [batch_size, encoder_len]
      attn_mask_flat = attn_mask[:, 0, :]

      # 1-pad attn_mask's token dimension (dim=1 of attn_mask_flat) to the full
      # token sequence length (length of n_tokens).
      # attn_mask is the length of unmasked_indices (encoded tokens), and thus
      # must be padded on the right to match the full sequence length.
      # Note that 1-padding represents 'present' in the attention mask.
      attn_mask_n_tokens = jnp.pad(
          attn_mask_flat,
          [(0, 0), (0, n_tokens - n_enc_tokens)],  # 1-pad axis=1
          mode='constant',
          constant_values=1,  # Padded with 1s (non-masked)
      )

      # Bit-flip attn_mask_n_tokens.
      # Want 1s to represent masked token to apply the mask token via jnp.where.
      # Shape attn_mask_flipped to be [batch, seq_len, 1]
      attn_mask_n_tokens_flipped = (attn_mask_n_tokens[:, :] == 0)[..., None]

      # Broadcast mask_token to x_all at masked positions (represeted by 1s)in
      # attn_mask_n_tokens_flipped.
      x_all = jnp.where(attn_mask_n_tokens_flipped, mask_token, x_all)

    # **** x_all is [Ea M M M Ee] ****
    # If there are tokens NOT passed to the encoder
    # (i.e. self.config.masker_config.strictmaskperc == 0) add mask_token at
    # these positions to x_all.
    # TODO(girishvn, xumax): Figure out a way to handle this conditioned on
    # mask_indices being None, or something similar (as opposed to a reference
    # to the config)
    if self.config.masker_config.strictmaskperc != 0.0:
      x_all = x_all.at[batch_indices, mask_indices].set(mask_token)

    x = x_all
    del x_all

    # Add positional encodings to the decoder.
    # x is shape [batch, time*modalities, self.decoder_config.hidden_size]
    if '1d' in self.positional_embedding_decoder:
      x = jnp.reshape(
          x, [batch, time, modalities, self.decoder_config.hidden_size]
      )
      newx_all = []
      for i in range(modalities):
        newx = mm_model_utils.add_positional_embeddings(
            x[:, :, i, :], self.positional_embedding_decoder
        )
        newx_all.append(newx[:, :, None, :])
      x = jnp.concatenate(newx_all, axis=2)
      assert x.shape == (
          batch, time, modalities, self.decoder_config.hidden_size,
      )
      x = jnp.reshape(x, [batch, n_tokens, self.decoder_config.hidden_size])
    else:
      x = mm_model_utils.add_positional_embeddings(
          x,
          self.positional_embedding_decoder,
          [batch, time, modalities, self.decoder_config.hidden_size],
      )

    # Decoder.
    # The parallel decoder, which is actually technically an encoder
    x = vit.Encoder(
        mlp_dim=self.decoder_config.mlp_dim,
        num_layers=self.decoder_config.num_layers,
        num_heads=self.decoder_config.num_heads,
        dropout_rate=self.decoder_config.dropout_rate,
        attention_dropout_rate=self.decoder_config.attention_dropout_rate,
        stochastic_depth=self.decoder_config.get('stochastic_depth', 0.0),
        dtype=self.dtype,
        positional_embedding='none',  # Has already been added.
        name='Decoder',
    )(x, train=train)

    # Predict pixel reconstructions.
    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    aux['pre_logits_decoder'] = x
    x = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection',
    )(x)

    return x, aux


# # enough time has passed, let's disable the misleading metrics
# (metric, normalizer, apply_prediction_weights)
_REGRESSION_METRICS = {
    # 'mean_squared_error_all': (
    #     functools.partial(mm_model_utils.weighted_error, loss_type='squared'),
    #     model_utils.num_examples,
    #     False,
    #     False,
    # ),
    # 'mean_absolute_error_all': (
    #     functools.partial(
    #         mm_model_utils.weighted_error, loss_type='absolute'
    # ),
    #     model_utils.num_examples,
    #     False,
    #     False,
    # ),
    # 'mean_squared_error_masked': (
    #     functools.partial(mm_model_utils.weighted_error, loss_type='squared'),
    #     model_utils.num_examples,
    #     True,
    #     False,
    # ),
    # 'mean_absolute_error_masked': (
    #     functools.partial(
    #         mm_model_utils.weighted_error, loss_type='absolute'
    #     ),
    #     model_utils.num_examples,
    #     True,
    #     False,
    # ),
    'mean_squared_error_masked_ignoreimp_mean': (
        functools.partial(
            mm_model_utils.weighted_error,
            loss_type='squared',
            # axis=tuple(range(1, 3)),  # ignore batch for now
            # mean=True,
        ),
        model_utils.num_examples,
        True,
        True,
    ),
    'mean_absolute_error_masked_ignoreimp_mean': (
        functools.partial(
            mm_model_utils.weighted_error,
            loss_type='absolute',
            # axis=tuple(range(1, 3)),  # ignore batch for now
            # mean=True,
        ),
        model_utils.num_examples,
        # LOSS_ONLY_MASKED_TOKENS, activating curr_weights*prediction_masks
        True,
        # LOSS_IGNORE_IMPUTATION, activating curr_weights*patched_imputationmask
        True,
    ),
}


def regression_metrics_function(
    predictions: jnp.ndarray,
    prediction_masks: jnp.ndarray,
    batch: base_model.Batch,
    metrics: base_model.MetricNormalizerFnDict,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, int]]:
  """Calculate metrics for the regression task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(predictions, targets, weights)```
  and returns an array of shape [batch_size,]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   predictions: Output of model in shape [batch_size, num_patches, patch_size].
     specifically, shape [batch_size, gh * gw, ph * pw]. see
     patchify_imputationmask func in trainers/lsm_mae_utils for more info ph and
     pw are the time and modalities of the patches (i.e. patch size) gh and gw
     are the total number of number of patches (i.e. num patches size)
   prediction_masks: Masks used for masked modeling, shape [batch_size,
     num_patches]
   batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
   metrics: The regression metrics to evaluate. The key is the name of the
     metric. The value is the metrics function, normalizer, a bool indicating
     whether to apply prediction_masks, and a bool indicating whether to apply
     patched_imputationmask
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  targets = batch['targets']
  batch_weights = batch.get('batch_mask')
  # create a mask with all data points, then chip at it based on input masks

  evaluated_metrics = {}
  for key, val in metrics.items():
    curr_weights = jnp.ones(targets.shape)

    # LOSS_ONLY_MASKED_TOKENS
    if val[2]:
      curr_weights = jnp.expand_dims(prediction_masks, axis=-1) * curr_weights
    # LOSS_IGNORE_IMPUTATION
    if val[3]:
      # see loss_function for ViTMAESingleChannelModel for a similar computation
      curr_weights = (
          jnp.logical_not(batch['patched_imputationmask']) * curr_weights
      )

    evaluated_metrics[key] = model_utils.psum_metric_normalizer(
        (
            val[0](
                targets,
                predictions,  # pytype: disable=wrong-arg-types  # jax-ndarray
                curr_weights,
            ),
            val[1](
                targets,
                predictions,  # pytype: disable=wrong-arg-types  # jax-ndarray
                batch_weights,
            ),
        ),
        axis_name=axis_name,
    )
  return evaluated_metrics  # pytype: disable=bad-return-type  # jax-ndarray


# no RGB so we will always use single channel
class ViTMAESingleChannelModel(base_model.BaseModel):
  """ViT-based masked modeling.

  Adapted from
  google3/third_party/py/scenic/projects/multimask/models/vit_mae.py.
  """

  def build_flax_model(self) -> nn.Module:
    model_dtype = getattr(jnp, self.config.get('model_dtype_str', 'float32'))
    num_classes = np.prod(tuple(self.config.model.patcher_config.patchsize)) * 1

    return ViTMAE(
        config=self.config,
        num_classes=num_classes,
        mlp_dim=self.config.model.mlp_dim,
        num_layers=self.config.model.num_layers,
        num_heads=self.config.model.num_heads,
        representation_size=self.config.model.representation_size,
        positional_embedding=self.config.model.positional_embedding,
        positional_embedding_decoder=self.config.model.positional_embedding_decoder,
        classifier='none',
        dropout_rate=self.config.model.get('dropout_rate', 0.1),
        attention_dropout_rate=self.config.model.get(
            'attention_dropout_rate', 0.1
        ),
        stochastic_depth=self.config.model.get('stochastic_depth', 0.0),
        decoder_config=self.config.model.decoder_config,
        dtype=model_dtype,
    )

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return ml_collections.ConfigDict()

  def init_from_train_state(
      self,
      train_state: Any,
      restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict,
  ) -> Any:
    """Updates the train_state with data from restored_train_state.

    This function is writen to be used for 'fine-tuning' experiments. Here, we
    do some surgery to support larger resolutions (longer sequence length) in
    the transformer block, with respect to the learned pos-embeddings.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a  pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    return vit.init_vit_from_train_state(
        train_state, restored_train_state, self.config, restored_model_cfg
    )

  # prediction_masks at the last position to fit the parent class func signature
  def loss_function(
      self,
      predictions: jnp.ndarray,
      batch: base_model.Batch,
      model_params: Optional[jnp.ndarray] = None,
      prediction_masks: Optional[jnp.ndarray] = None,
      loss_ignore_imputation: bool = False,
  ) -> float:
    """Returns the (weighted) mean squared error.

    Args:
      predictions: Output of model in shape [batch_size, num patches, patch
        size]. specifically, shape [batch_size, gh * gw, ph * pw]. see
        patchify_imputationmask func in trainers/lsm_mae_utils for more info ph
        and pw are the time and modalities of the patches (i.e. patch size) gh
        and gw are the total number of number of patches (i.e. num patches size)
      batch: Batch (dict) with keys 'targets' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.
      prediction_masks: Masks used for masked modeling, shape [batch_size, num
        patches] where 1 is "masked" (this is also called token_mask)
      loss_ignore_imputation: Whether to use masked loss or not.

    Returns:
      The scalar loss, which is the (weighted) absolute error.
    """
    # IIUC, this mask can be provided by the data loader to indicate invalid
    # examples, e.g. for incomplete batches during eval when
    # drop_remainder=False on the final batch, otherwise, it is a ones array
    curr_weights = jnp.ones(predictions.shape)

    # If requested, compute the loss only on masked tokens
    if self.config.masked_feature_loss.loss_only_masked_tokens:
      curr_weights = jnp.expand_dims(prediction_masks, axis=-1) * curr_weights


    # shape [batch, num patches, patch size]
    targets = batch['targets']
    if loss_ignore_imputation:
      # 1 is imputed and 0 is non-imputed
      patched_imputationmask = batch['patched_imputationmask']
      assert predictions.shape == patched_imputationmask.shape
      assert targets.shape == patched_imputationmask.shape

      curr_weights = jnp.logical_not(patched_imputationmask) * curr_weights

    # total_loss has shape [batch_size]
    total_loss = mm_model_utils.weighted_error(
        predictions,
        targets,
        curr_weights,
        axis=tuple(range(1, targets.ndim)),  # ignore batch for now
        loss_type=self.config.masked_feature_loss.loss_type,
        mean=True,  # avgs over total num of patches via axis set
    )
    # apply batch mask
    total_loss = total_loss * batch['batch_mask']
    total_loss = jnp.sum(total_loss) / jnp.sum(batch['batch_mask'])

    return total_loss  # pytype: disable=bad-return-type  # jax-ndarray

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    By default, we return the same metric for each split.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].

    Returns: A metric function with the following API:
      ```metrics_fn(predictions, batch)```
    """

    del split  # Same function for all splits.
    return functools.partial(
        regression_metrics_function, metrics=_REGRESSION_METRICS
    )
