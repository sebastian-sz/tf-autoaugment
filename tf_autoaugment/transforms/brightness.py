"""Code for Brightness Transform."""
import tensorflow as tf

from tf_autoaugment.image_utils import blend_batch
from tf_autoaugment.transforms.base_transform import BaseTransform


class Brightness(BaseTransform):
    """Implements Brightness Transform."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run brightness function on image batch."""
        factor = self._parse_level(level)
        return self.brightness(images=images, factor=factor)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 1.8 + 0.1
        return tf.cast(result, tf.float32)

    @staticmethod
    def brightness(images: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
        """Equivalent of PIL Brightness."""
        degenerate = tf.zeros_like(images)
        return blend_batch(degenerate, images, factor)
