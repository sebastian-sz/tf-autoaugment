"""Code for Color transform."""
import tensorflow as tf

from tf_autoaugment.image_utils import blend_batch
from tf_autoaugment.transforms.base_transform import BaseTransform


class Color(BaseTransform):
    """Implements Color Transform."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Parse level and run color function on image batch."""
        factor = self._parse_level(level)
        return self.color(images=images, factor=factor)

    @staticmethod
    def color(images: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
        """Adjust the color balance of the image."""
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(images))
        return blend_batch(degenerate, images, factor)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 1.8 + 0.1
        return tf.cast(result, tf.float32)
