"""Code for Sharpness Transform."""
import tensorflow as tf

from tf_autoaugment.image_utils import blend_batch
from tf_autoaugment.transforms.base_transform import BaseTransform


class Sharpness(BaseTransform):
    """Implements Sharpness transform."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run sharpness on image batch."""
        factor = self._parse_level(level)
        return self.sharpness(images=images, factor=factor)

    @staticmethod
    def sharpness(images: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
        """Sharpness copied from TFA, without tf.function."""
        orig_image = images
        image_dtype = images.dtype
        image_channels = images.shape[-1]
        images = tf.cast(images, tf.float32)

        # SMOOTH PIL Kernel.
        kernel = (
            tf.constant(
                [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
            )
            / 13.0
        )
        kernel = tf.tile(kernel, [1, 1, image_channels, 1])

        # Apply kernel channel-wise.
        degenerate = tf.nn.depthwise_conv2d(
            images, kernel, strides=[1, 1, 1, 1], padding="VALID", dilations=[1, 1]
        )
        degenerate = tf.cast(degenerate, image_dtype)

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(degenerate)
        padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
        padded_degenerate = tf.pad(degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])
        result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

        # Blend the final result.
        blended = blend_batch(result, orig_image, factor)
        return tf.cast(blended, image_dtype)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 1.8 + 0.1
        return tf.cast(result, tf.float32)
