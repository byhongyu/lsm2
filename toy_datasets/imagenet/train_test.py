import functools
import time
from unittest import mock

from absl import logging
from absl.testing import parameterized
from etils import epath
import grain.tensorflow as tf_grain
import jax
import jax.numpy as jnp
import tensorflow as tf

from google3.experimental.largesensormodels.toy_datasets.imagenet import train  # pylint: disable=fine-too-long CHANGEME
from google3.experimental.largesensormodels.toy_datasets.imagenet.configs import default  # pylint: disable=fine-too-long CHANGEME


class TrainTest(tf.test.TestCase, parameterized.TestCase):
  """Test cases for ImageNet library."""

  def setUp(self):
    super().setUp()
    tf.config.experimental.set_visible_devices([], "GPU")
    self.tmp_dir = epath.Path(f"/tmp/workdir_{int(time.time())}")

  def tearDown(self):
    if self.tmp_dir.exists():
      self.tmp_dir.rmtree()
    super().tearDown()

  def test_train_and_evaluate(self):
    config = default.get_config()
    config.model_name = "resnet18"
    config.per_device_batch_size = 1
    config.num_train_steps = 1
    config.num_eval_steps = 1
    config.num_epochs = 1
    config.warmup_epochs = 0

    encoded_data_source = tf_grain.TfInMemoryDataSource({
        # Fake encoded image for the train set.
        "image": tf.constant(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x03\x02\x02\x02\x02\x02\x03\x02\x02\x02\x03\x03\x03\x03\x04\x06\x04\x04\x04\x04\x04\x08\x06\x06\x05\x06\t\x08\n\n\t\x08\t\t\n\x0c\x0f\x0c\n\x0b\x0e\x0b\t\t\r\x11\r\x0e\x0f\x10\x10\x11\x10\n\x0c\x12\x13\x12\x10\x13\x0f\x10\x10\x10\xff\xc9\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xcc\x00\x06\x00\x10\x10\x05\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf"
            b" \xff\xd9",
            dtype=tf.string,
            shape=(64,),
        ),
        "label": tf.reshape(tf.range(64, dtype=tf.int32), [64]),
        "file_name": tf.constant("name", dtype=tf.string, shape=(64,)),
    })
    decoded_data_source = tf_grain.TfInMemoryDataSource({
        # Fake decoded image for the eval set.
        "image": tf.ones([64, 28, 28, 3], dtype=tf.float32),
        "label": tf.reshape(tf.range(64, dtype=tf.int32), [64]),
        "file_name": tf.constant("name", dtype=tf.string, shape=(64,)),
    })

    with (
        mock.patch.object(
            tf_grain.TfdsDataSource,
            "from_name",
            side_effect=[encoded_data_source, decoded_data_source] * 2,
        ),
        mock.patch.object(
            jax.profiler,
            "StepTraceAnnotation",
            wraps=jax.profiler.StepTraceAnnotation,
        ) as step_trace_annotation,
    ):
      checkpoint = train.train_and_evaluate(config, self.tmp_dir)
      state = train.load_last_state(self.tmp_dir, checkpoint)
      # We did enter once in the training loop:
      step_trace_annotation.assert_has_calls(
          [mock.call("train", step_num=1), mock.call("eval", step_num=0)]
      )
      self.assertEqual(state.step, 1)  # `step` went from 0 to 1.

      # Trying a second training loop:
      train.train_and_evaluate(config, self.tmp_dir)
      state = train.load_last_state(self.tmp_dir, checkpoint)
      # Checkpointing works, because we didn't make a second training loop:
      step_trace_annotation.assert_has_calls(
          [mock.call("train", step_num=1), mock.call("eval", step_num=0)]
      )
      # `step` did not change, as training is over.
      self.assertEqual(state.step, 1)

  @parameterized.parameters(
      (0, 0.0),  #
      (1, 6.410256901290268e-05),  #
      (1000, 0.06410256773233414),  #
      (1560, 0.10000000149011612),  #
      (3000, 0.09927429258823395),  #
      (6000, 0.09324192255735397),  #
      (10000, 0.077022984623909),
  )
  def test_get_learning_rate(self, step: int, expected_lr: float):
    actual_lr = train.get_learning_rate(
        step, base_learning_rate=0.1, steps_per_epoch=312, num_epochs=90
    )
    self.assertAllClose(expected_lr, actual_lr)

  @parameterized.parameters(
      (0, 0.0),  #
      (1, 6.410256901290268e-05),  #
      (1000, 0.06410256773233414),  #
      (1560, 0.10000000149011612),  #
      (3000, 0.09927429258823395),  #
      (6000, 0.09324192255735397),  #
      (10000, 0.077022984623909),
  )
  def test_get_learning_rate_jitted(self, step: int, expected_lr: float):
    lr_fn = jax.jit(
        functools.partial(
            train.get_learning_rate,
            base_learning_rate=0.1,
            steps_per_epoch=312,
            num_epochs=90,
        )
    )
    actual_lr = lr_fn(jnp.array(step))
    self.assertAllClose(expected_lr, actual_lr)

  def test_evaluate(self):
    per_device_batch_size = 2
    process_batch_size = jax.local_device_count() * per_device_batch_size
    eval_ds = tf.data.Dataset.from_tensors(
        dict(
            image=tf.zeros(shape=(process_batch_size, 28, 28, 1)),
            label=tf.constant(jax.local_device_count() * [0, 9]),
        )
    )

    logits = (jnp.arange(10.0).reshape(1, -1),)

    class MockedState:

      def __init__(self):
        self.params = {}
        self.batch_stats = {}

    jax.tree_util.register_pytree_node(
        MockedState, lambda _: ((), None), lambda *_: MockedState()
    )

    model = mock.Mock()
    model.apply.side_effect = logits

    eval_metrics = train.evaluate(model, MockedState(), eval_ds)
    logging.info("eval_metrics: %s", eval_metrics.compute())
    self.assertAllClose(
        {
            "accuracy": 0.5,
            "eval_loss": 4.9586296,
        },
        eval_metrics.compute(),
        # Lower precision to cover both single-device GPU and
        # multi-device TPU loss that are slightly different.
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == "__main__":
  tf.test.main()
