"""Code for Autocontrast Transform."""

import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform


class AutoContrast(BaseTransform):
    """Implements Autocontrast Transform."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run autocontrast function on image batch."""
        return self.autocontrast(images)

    def autocontrast(self, images: tf.Tensor) -> tf.Tensor:
        """Wrap autocontrast in graph friendly for-loop."""
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

            transformed_frame = self._autocontrast(single_tensor)
            transformed_frame = tf.cast(transformed_frame, original_dtype)

            results = results.write(batch_id, transformed_frame)

        return results.stack()

    # Todo: add source.
    @staticmethod
    def _autocontrast(image):
        def scale_channel(image):
            """Scale the 2D image using the autocontrast rule."""
            # A possibly cheaper version can be done using cumsum/unique_with_counts
            # over the histogram values, rather than iterating over the entire image.
            # to compute mins and maxes.
            lo = tf.cast(tf.reduce_min(image), tf.float32)
            hi = tf.cast(tf.reduce_max(image), tf.float32)

            # Scale the image, making the lowest value 0 and the highest value 255.
            def scale_values(im):
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                im = tf.cast(im, tf.float32) * scale + offset
                im = tf.clip_by_value(im, 0.0, 255.0)
                return tf.cast(im, tf.uint8)

            result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
            return result

        # Scale and stack each channel independently.
        s1 = scale_channel(image[:, :, 0])
        s2 = scale_channel(image[:, :, 1])
        s3 = scale_channel(image[:, :, 2])
        image = tf.stack([s1, s2, s3], 2)
        return image

    def _parse_level(self, level: int) -> None:
        """Level unused by this op."""
        pass
