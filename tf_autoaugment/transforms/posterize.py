"""Code for Posterize Transform."""
import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform


class Posterize(BaseTransform):
    """Transform for Posterize op."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run posterize function on image batch."""
        original_dtype = images.dtype
        bits = self._parse_level(level)
        inputs = tf.cast(images, tf.uint8)
        outputs = self.posterize(images=inputs, bits=bits)
        return tf.cast(outputs, original_dtype)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 4
        return tf.cast(result, tf.uint8)

    @staticmethod
    def posterize(images: tf.Tensor, bits: tf.Tensor) -> tf.Tensor:
        """Reduce the number of bits for each pixel."""
        shift = 8 - bits
        return tf.bitwise.left_shift(tf.bitwise.right_shift(images, shift), shift)
