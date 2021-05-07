"""Code for Contrast Transform."""
import tensorflow as tf

from tf_autoaugment.image_utils import blend_batch
from tf_autoaugment.transforms.base_transform import BaseTransform


class Contrast(BaseTransform):
    """Implements Contrast Transform."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run contrast function on image batch."""
        factor = self._parse_level(level)
        return self.contrast(images=images, factor=factor)

    def contrast(self, images: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
        """Contrast wrapped in graph-friendly for loop."""
        batch_size = images.shape[0]
        original_dtype = images.dtype
        results = tf.TensorArray(
            dtype=original_dtype,
            size=batch_size,
            element_shape=tf.TensorShape(dims=images[0].shape),
        )

        for batch_id in tf.range(batch_size):
            single_tensor = tf.gather(images, batch_id)
            single_tensor = tf.cast(single_tensor, tf.uint8)

            transformed_frame = self._contrast(single_tensor, factor)
            transformed_frame = tf.cast(transformed_frame, original_dtype)

            results = results.write(batch_id, transformed_frame)

        return results.stack()

    @staticmethod
    def _contrast(image: tf.Tensor, factor: tf.Tensor) -> tf.Tensor:
        """Equivalent of PIL Contrast."""
        degenerate = tf.image.rgb_to_grayscale(image)
        # Cast before calling tf.histogram.
        degenerate = tf.cast(degenerate, tf.int32)

        # Compute the grayscale histogram, then compute the mean pixel value,
        # and create a constant image size of that value.  Use that as the
        # blending degenerate target of the original image.
        hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
        return blend_batch(degenerate, image, factor)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 1.8 + 0.1
        return tf.cast(result, tf.float32)
