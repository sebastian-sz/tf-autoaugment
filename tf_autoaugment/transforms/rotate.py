"""Code for Rotate Transform."""
import tensorflow as tf
import tensorflow_addons as tfa

from tf_autoaugment.transforms.base_transform import BaseTransform


class Rotate(BaseTransform):
    """Transform for Rotate op."""

    def __init__(self, fill_value=128, tf_function: bool = False):
        super().__init__(tf_function=tf_function)

        self.fill_value = fill_value

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run rotate on image batch."""
        angles = self._parse_level(level)
        return tfa.image.rotate(
            images=images, angles=angles, fill_value=self.fill_value
        )

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        level = (level / self._MAX_LEVEL) * 30.0
        level = self._randomly_negate_tensor(tf.convert_to_tensor(level))
        return tf.cast(level, tf.float32)

    def _debug_run(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        angles = tf.cast((level / self._MAX_LEVEL) * 30.0, tf.float32)
        return tfa.image.rotate(
            images=images, angles=angles, fill_value=self.fill_value
        )
