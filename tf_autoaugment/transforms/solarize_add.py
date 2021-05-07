"""Code for Solarize Add Transform."""
import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform

# Todo: double check this op. Not sure what the threshold should be here.


class SolarizeAdd(BaseTransform):
    """Implements Solarize Add Transform."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run solarize add function on image batch."""
        addition = self._parse_level(level)
        return self.solarize_add(
            images=images, addition=addition
        )  # Todo: make sure this is ok. What value for threshold?

    @staticmethod
    def solarize_add(
        images: tf.Tensor, addition: tf.Tensor = 0, threshold: tf.Tensor = 128
    ) -> tf.Tensor:
        """Perform solarization with adding.

        For each pixel in the image less than threshold we add 'addition'
        amount to it and then clip the pixel value to be between 0 and 255. The value
        of 'addition' is between -128 and 128.
        """
        original_dtype = images.dtype

        images = tf.cast(images, tf.uint8)
        added_image = tf.cast(images, tf.int64) + addition
        added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
        result = tf.where(images < threshold, added_image, images)

        return tf.cast(result, original_dtype)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 110
        return tf.cast(result, tf.int64)
