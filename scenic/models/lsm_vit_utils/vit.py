"""Modified vit pulled from third_party/py/scenic/projects/baselines/vit.py

Everything is basically copy and pasted exactly except for the attn_mask pass
through.
"""

from typing import Any, Callable, Optional, Sequence

from absl import logging
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.layers import attention_layers
from scenic.model_lib.layers import nn_layers
import scipy

Initializer = Callable[[jnp.ndarray, Sequence[int], jnp.dtype], jnp.ndarray]


class ViT(nn.Module):
  """Vision Transformer model.

  Attributes:

  num_classes: Number of output classes.
  mlp_dim: Dimension of the mlp on top of attention block.
  num_layers: Number of layers.
  num_heads: Number of self-attention heads.
  patches: Configuration of the patches extracted in the stem of the model.
  hidden_size: Size of the hidden state of the output of model's stem.
  positional_embedding: The type of positional embeddings to add to the
    tokens at the beginning of the transformer encoder. Options are
    {learned_1d, sinusoidal_2d, none}.
  representation_size: Size of the representation layer in the model's head.
    if None, we skip the extra projection + tanh activation at the end.
  dropout_rate: Dropout rate.
  attention_dropout_rate: Dropout for attention heads.
  classifier: type of the classifier layer. Options are 'gap', 'gmp', 'gsp',
    'token', 'none'.
  dtype: JAX data type for activations.
  """

  num_classes: int
  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  positional_embedding: str = 'learned_1d'
  representation_size: Optional[int] = None
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  classifier: str = 'gap'
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(
      self, x: jnp.ndarray, *, train: bool, attn_mask=None, debug: bool = False
  ):

    fh, fw = self.patches.size
    # Extracting patches and then embedding is in fact a single convolution.
    x = nn.Conv(
        self.hidden_size,
        (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding',
    )(x)
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        positional_embedding=self.positional_embedding,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        stochastic_depth=self.stochastic_depth,
        dtype=self.dtype,
        has_cls_token=self.classifier == 'token',
        name='Transformer',
    )(
        x, attn_mask=attn_mask, train=train
    )

    if self.classifier in ('token', '0'):
      x = x[:, 0]
    elif self.classifier in ('gap', 'gmp', 'gsp'):
      fn = {'gap': jnp.mean, 'gmp': jnp.max, 'gsp': jnp.sum}[self.classifier]
      x = fn(x, axis=1)
    elif self.classifier == 'map':
      x = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim, dtype=self.dtype
      )(x)
    elif self.classifier == 'none':
      pass
    else:
      raise ValueError(f'Unknown classifier {self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = nn_layers.IdentityLayer(name='pre_logits')(x)

    if self.num_classes > 0:
      # If self.num_classes <= 0, we just return the backbone features.
      x = nn.Dense(
          self.num_classes,
          kernel_init=nn.initializers.zeros,
          name='output_projection',
      )(x)
    return x


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""

  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    n, _, d = x.shape
    probe = self.param(
        'probe', nn.initializers.xavier_uniform(), (1, 1, d), x.dtype
    )
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform()
    )(probe, x)

    y = nn.LayerNorm()(x)
    x = x + attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=0.0
    )(y, deterministic=True)
    return x[:, 0]


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: The number of heads for multi-head self-attention.
    positional_embedding: The type of positional embeddings to add to the input
      tokens. Options are {learned_1d, sinusoidal_2d, none}.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows timm
      library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
    has_cls_token: Whether or not the sequence is prepended with a CLS token.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  positional_embedding: str = 'learned_1d'
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32
  has_cls_token: bool = False

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, *, attn_mask=None, train: bool = False
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Input tokens of shape [batch, num_tokens, channels].
      train: If in training mode, dropout and stochastic depth is applied.

    Returns:
      Encoded tokens.
    """

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    # Add positional embeddings to tokens.
    if self.positional_embedding == 'learned_1d':
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input',
      )(inputs)
    elif self.positional_embedding == 'sinusoidal_1d':
      x = attention_layers.Add1DPositionEmbedding(posemb_init=None)(inputs)
    elif self.positional_embedding == 'sinusoidal_2d':
      batch, num_tokens, hidden_dim = inputs.shape
      if self.has_cls_token:
        num_tokens -= 1
      height = width = int(np.sqrt(num_tokens))
      if height * width != num_tokens:
        raise ValueError('Input is assumed to be square for sinusoidal init.')
      if self.has_cls_token:
        inputs_reshape = inputs[:, 1:].reshape(
            [batch, height, width, hidden_dim]
        )
        x = attention_layers.AddFixedSinCosPositionEmbedding()(inputs_reshape)
        x = x.reshape([batch, num_tokens, hidden_dim])
        x = jnp.concatenate([inputs[:, :1], x], axis=1)
      else:
        inputs_reshape = inputs.reshape([batch, height, width, hidden_dim])
        x = attention_layers.AddFixedSinCosPositionEmbedding()(inputs_reshape)
        x = x.reshape([batch, num_tokens, hidden_dim])
    elif self.positional_embedding == 'none':
      x = inputs
    else:
      raise ValueError(
          f'Unknown positional embedding: {self.positional_embedding}'
      )
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder.
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1))
          * self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype,
      )(x, deterministic=not train, attn_mask=attn_mask)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """

  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, (
        'Number of dimensions should be 3, but it is: %d' % inputs.ndim
    )
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param(
        'pos_embedding', self.posemb_init, pos_emb_shape, inputs.dtype
    )
    return inputs + pe


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """

  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0

  @nn.compact
  def __call__(
      self, inputs: jnp.ndarray, deterministic: bool, attn_mask=None
  ) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
    )(
        x,
        x,
        mask=attn_mask,
    )
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = nn_layers.StochasticDepth(rate=self.stochastic_depth)(x, deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = attention_layers.MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )(y, deterministic=deterministic)
    y = nn_layers.StochasticDepth(rate=self.stochastic_depth)(y, deterministic)
    return y + x


def init_vit_from_train_state(
    train_state: Any,
    restored_train_state: Any,
    model_cfg: ml_collections.ConfigDict,
    restored_model_cfg: ml_collections.ConfigDict,
) -> Any:
  """Updates the train_state with data from restored_train_state.

  This function is written to be used for 'fine-tuning' experiments. Here, we
  do some surgery to support larger resolutions (longer sequence length) in
  the transformer block, with respect to the learned pos-embeddings.

  The function supports train_states using either Optax or flax.optim (which
  has been deprecated, and will be removed from Scenic.)

  Args:
    train_state: A raw TrainState for the model.
    restored_train_state: A TrainState that is loaded with parameters/state of a
      pretrained model.
    model_cfg: Configuration of the model. Usually used for some asserts.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.

  Returns:
    Updated train_state.
  """
  if hasattr(train_state, 'optimizer'):
    # TODO(dehghani): Remove support for flax optim.
    params = flax.core.unfreeze(train_state.optimizer.target)
    restored_params = flax.core.unfreeze(restored_train_state.optimizer.target)
    _merge_params(params, restored_params, model_cfg, restored_model_cfg)
    return train_state.replace(
        optimizer=train_state.optimizer.replace(target=flax.core.freeze(params))
    )
  else:
    params = flax.core.unfreeze(train_state.params)
    restored_params = flax.core.unfreeze(restored_train_state.params)
    _merge_params(params, restored_params, model_cfg, restored_model_cfg)
    return train_state.replace(params=flax.core.freeze(params))


def _merge_params(params, restored_params, model_cfg, restored_model_cfg):
  """Merges `restored_params` into `params`."""
  # Start moving parameters, one-by-one and apply changes if needed.
  for m_key, m_params in restored_params.items():
    if m_key == 'output_projection':
      # For the classifier head, we use a the randomly initialized params and
      #   ignore the one from pretrained model.
      pass

    elif m_key == 'pre_logits':
      if model_cfg.model.representation_size is None:
        # We don't have representation_size in the new model, so let's ignore
        #   it from the pretained model, in case it has it.
        # Note, removing the key from the dictionary is necessary to prevent
        #   obscure errors from the Flax optimizer.
        params.pop(m_key, None)
      else:
        assert restored_model_cfg.model.representation_size
        params[m_key] = m_params

    elif m_key == 'Transformer':
      for tm_key, tm_params in m_params.items():
        if tm_key == 'posembed_input':  # Might need resolution change.
          posemb = params[m_key]['posembed_input']['pos_embedding']
          restored_posemb = m_params['posembed_input']['pos_embedding']

          if restored_posemb.shape != posemb.shape:
            # Rescale the grid of pos, embeddings: param shape is (1, N, d).
            logging.info(
                'Resized variant: %s to %s', restored_posemb.shape, posemb.shape
            )
            ntok = posemb.shape[1]
            if restored_model_cfg.model.classifier == 'token':
              # The first token is the CLS token.
              restored_posemb_grid = restored_posemb[0, 1:]
              if model_cfg.model.classifier == 'token':
                # CLS token in restored model and in target.
                cls_tok = restored_posemb[:, :1]
                ntok -= 1
              else:
                # CLS token in restored model, but not target.
                cls_tok = restored_posemb[:, :0]
            else:
              restored_posemb_grid = restored_posemb[0]
              if model_cfg.model.classifier == 'token':
                # CLS token in target, but not restored model.
                cls_tok = posemb[:, :1]
                ntok -= 1
              else:
                # CLS token not in target or restored model.
                cls_tok = restored_posemb[:, :0]

            restored_gs = int(np.sqrt(len(restored_posemb_grid)))
            gs = int(np.sqrt(ntok))
            if restored_gs != gs:  # We need resolution change.
              logging.info('Grid-size from %s to %s.', restored_gs, gs)
              restored_posemb_grid = restored_posemb_grid.reshape(
                  restored_gs, restored_gs, -1
              )
              zoom = (gs / restored_gs, gs / restored_gs, 1)
              restored_posemb_grid = scipy.ndimage.zoom(
                  restored_posemb_grid, zoom, order=1
              )
            # Attach the CLS token again.
            restored_posemb_grid = restored_posemb_grid.reshape(1, gs * gs, -1)
            restored_posemb = jnp.array(
                np.concatenate([cls_tok, restored_posemb_grid], axis=1)
            )

          params[m_key][tm_key]['pos_embedding'] = restored_posemb
        # Other parameters of the Transformer encoder if they are in the target.
        elif tm_key in params[m_key]:
          params[m_key][tm_key] = tm_params
        else:
          logging.info(
              "Ignoring %s. In restored model's Transformer,but not in target",
              m_key,
          )

    elif m_key in params:
      # Use the rest if they are in the pretrained model.
      params[m_key] = m_params

    else:
      logging.info('Ignoring %s. In restored model, but not in target', m_key)
