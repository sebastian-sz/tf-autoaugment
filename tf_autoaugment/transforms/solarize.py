"""Code for Solarize Transform."""
import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform


class Solarize(BaseTransform):
    """Transform for Solarize op."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Call solarize on input images."""
        threshold = self._parse_level(level)
        threshold = tf.cast(threshold, images.dtype)
        return self.solarize(images=images, threshold=threshold)

    @staticmethod
    def solarize(images: tf.Tensor, threshold: tf.Tensor) -> tf.Tensor:
        """Invert all pixels above a threshold value."""
        return tf.where(images < threshold, images, 255 - images)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 256
        return tf.cast(result, tf.int32)
