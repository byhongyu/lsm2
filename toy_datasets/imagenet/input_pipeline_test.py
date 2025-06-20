import tensorflow as tf

from google3.experimental.largesensormodels.toy_datasets.imagenet import input_pipeline


class InputPipelineTest(tf.test.TestCase):

  def test_resize_small(self):
    image = tf.ones([8, 4, 3])
    self.assertAllEqual([6, 3, 3],
                        input_pipeline.ResizeSmall(3).map({"image": image
                                                          })["image"].shape)
    image = tf.ones([5, 15, 3])
    self.assertAllEqual([3, 9, 3],
                        input_pipeline.ResizeSmall(3).map({"image": image
                                                          })["image"].shape)
    self.assertAllEqual([10, 30, 3],
                        input_pipeline.ResizeSmall(10).map({"image": image
                                                           })["image"].shape)

  def test_central_crop(self):
    image = tf.reshape(tf.range(9) + 1, [3, 3, 1])
    image = tf.pad(image, [[3, 1], [3, 3], [0, 0]])
    image_actual = input_pipeline.CentralCrop(3).map({"image": image})["image"]
    self.assertEqual([3, 3, 1], image_actual.shape)
    image_expected = tf.constant([[[0], [0], [0]], [[1], [2], [3]],
                                  [[4], [5], [6]]])
    self.assertAllEqual(image_expected, image_actual)


if __name__ == "__main__":
  tf.test.main()
