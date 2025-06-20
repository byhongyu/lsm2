"""Deterministic input pipeline for ImageNet.

Training and evaluation preprocessing is hardcoded for simplicity and
readability. If experimenting with perprocessing, take a look at CLU:
  google3/third_party/py/clu/preprocess_spec.py.
This enables definition of preprocessing ops and construction of preprocessing
functions from simple strings (which could be part of an experimental config and
changed in hyperparameter sweeps).
"""
import dataclasses
from typing import Sequence, Tuple

from clu import preprocess_spec
import grain.tensorflow as tf_grain
import jax
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

FlatFeatures = preprocess_spec.FlatFeatures


@dataclasses.dataclass(frozen=True)
class ResizeSmall(tf_grain.MapTransform):
  """Resizes the smaller side to `size` keeping aspect ratio.

  Attr:
    size: Smaller side of an input image (might be adjusted if max_size given).
    antialias: Whether to use an anti-aliasing filter when downsampling an
      image.
  """

  size: int
  antialias: bool = False

  def map(self, features: FlatFeatures) -> FlatFeatures:
    image = features["image"]
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    # Figure out the necessary h/w.
    ratio = (
        tf.cast(self.size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
    features["image"] = tf.image.resize(image, [h, w], antialias=self.antialias)
    return features


@dataclasses.dataclass(frozen=True)
class CentralCrop(tf_grain.MapTransform):
  """Makes a central crop of a given size.

  Attributes:
    size: integer side length for (square) images.
  """

  size: int

  def map(self, features: FlatFeatures) -> FlatFeatures:
    image = features["image"]
    h, w = self.size, self.size
    top = (tf.shape(image)[0] - h) // 2
    left = (tf.shape(image)[1] - w) // 2
    features["image"] = tf.image.crop_to_bounding_box(image, top, left, h, w)
    return features


@dataclasses.dataclass(frozen=True)
class DecodeAndRandomResizedCrop(tf_grain.RandomMapTransform):
  """Decodes an image and extracts a random crop.

  Attributes:
    resize_size: integer side length for (square) images, produced by resizing
      after cropping.
  """

  resize_size: int

  def random_map(self, features: FlatFeatures, seed: tf.Tensor) -> FlatFeatures:
    image = features["image"]
    shape = tf.io.extract_jpeg_shape(image)
    begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        seed=seed,
        area_range=(0.05, 1.0),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    top, left, _ = tf.unstack(begin)
    h, w, _ = tf.unstack(size)
    image = tf.image.decode_and_crop_jpeg(image, [top, left, h, w], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    features["image"] = tf.image.resize(image,
                                        (self.resize_size, self.resize_size))
    return features


@dataclasses.dataclass(frozen=True)
class RandomFlipLeftRight(tf_grain.RandomMapTransform):
  """Randomly flips images horizontally."""

  def random_map(self, features: FlatFeatures, seed: tf.Tensor) -> FlatFeatures:
    features["image"] = tf.image.stateless_random_flip_left_right(
        features["image"], seed)
    return features


@dataclasses.dataclass(frozen=True)
class RescaleValues(tf_grain.MapTransform):
  """Rescales values from `min/max_input` to `min/max_output`.

  Attr:
    min_output: The minimum value of the output.
    max_output: The maximum value of the output.
    min_input: The minimum value of the input.
    max_input: The maximum value of the input.
    clip: Whether to clip the output value, in case of input out-of-bound.
  """

  min_output: float = 0.
  max_output: float = 1.
  min_input: float = 0.
  max_input: float = 255.0

  def __post_init__(self):
    assert self.min_output < self.max_output
    assert self.min_input < self.max_input

  def map(self, features: FlatFeatures) -> FlatFeatures:
    image = features["image"]
    min_input = tf.constant(self.min_input, tf.float32)
    max_input = tf.constant(self.max_input, tf.float32)
    image = tf.cast(image, tf.float32)
    image = (image - min_input) / (max_input - min_input)
    image = self.min_output + image * (self.max_output - self.min_output)
    features["image"] = image
    return features


@dataclasses.dataclass(frozen=True)
class DropFeatures(tf_grain.MapTransform):
  """Drops specified features from the tensor dictionary.

  Attributes:
    feature_names: Sequence of strings indicating which keys to drop.
  """
  feature_names: Sequence[str]

  def map(self, features: FlatFeatures) -> FlatFeatures:
    return {k: v for k, v in features.items() if k not in self.feature_names}


def get_num_train_steps(config: ml_collections.ConfigDict) -> int:
  """Calculates the total number of training steps."""
  if config.num_train_steps > 0:
    return config.num_train_steps
  # from the beginning. We first shard the data (shard by process_count), then
  # combine all epochs, batch for all local devices.
  # In all steps we would drop the remainder (hence the use of integer
  # devision).
  # When start_index is 0 the train_ds.cardinality() and num_train_steps should
  # be equavilent.
  tfds_info = tfds.builder("imagenet2012").info
  num_train_records = tfds_info.splits["train"].num_examples
  return (int(num_train_records // jax.process_count() * config.num_epochs) //
          (config.per_device_batch_size * jax.local_device_count()))


def create_datasets(
    config: ml_collections.ConfigDict,
    seed: int,
) -> Tuple[tf_grain.TfDataLoader, tf_grain.TfDataLoader]:
  """Create tf_grain data loaders for training and evaluation.

  For the same seed and config this will return the same datasets.
  The user is responsible to save()/load() the dataset iterators (for training)
  or calling reset() to restart the iterator (for eval).

  Args:
    config: Configuration to use.
    seed: Seed for shuffle and random operations in the training dataset.

  Returns:
    A tuple with the training dataset loader and the evaluation dataset
    loader.
  """
  # The input pipeline runs on each process and loads data for local TPUs.
  process_batch_size = jax.local_device_count() * config.per_device_batch_size

  # This is currently required to make the input pipeline fast enough on
  # Pufferfish machines. For many larger models this can be removed.
  tf_grain.config.update("tf_interleaved_shuffle", True)

  train_transformations = [
      DecodeAndRandomResizedCrop(resize_size=224),
      RandomFlipLeftRight(),
      DropFeatures(("file_name",))
  ]
  train_decoders = {"image": tfds.decode.SkipDecoding()}
  train_loader = tf_grain.load_from_tfds(
      name="imagenet2012",
      # TODO(b/240535786): Remove data_dir once ArrayRecord is the default.
      data_dir=tfds.core.constants.ARRAY_RECORD_DATA_DIR,
      split="train",
      shuffle=True,
      seed=seed,
      shard_options=tf_grain.ShardByJaxProcess(drop_remainder=True),
      decoders=train_decoders,
      transformations=train_transformations,
      batch_size=process_batch_size,
  )

  eval_transformations = [
      RescaleValues(),
      ResizeSmall(size=256),
      CentralCrop(224),
      DropFeatures(("file_name",))
  ]
  if config.eval_pad_last_batch:
    shard_options = tf_grain.ShardByJaxProcess()
    eval_batch_fn = tf_grain.TfBatchWithPadElements(
        process_batch_size, mask_key="mask")
  else:
    shard_options = tf_grain.ShardByJaxProcess(drop_remainder=True)
    eval_batch_fn = tf_grain.TfBatch(process_batch_size, drop_remainder=True)
  eval_loader = tf_grain.load_from_tfds(
      name="imagenet2012",
      # TODO(b/240535786): Remove data_dir once ArrayRecord is the default.
      data_dir=tfds.core.constants.ARRAY_RECORD_DATA_DIR,
      split="validation",
      num_epochs=1,
      shard_options=shard_options,
      transformations=eval_transformations,
      batch_fn=eval_batch_fn,
  )

  return train_loader, eval_loader
