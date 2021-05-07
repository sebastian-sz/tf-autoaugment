"""Code for Equalize Transform."""
import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform


class Equalize(BaseTransform):
    """Transform for Equalize op."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run equalize on image batch."""
        return self.equalize(images=images)

    def equalize(self, images: tf.Tensor):
        """Equalize wrapped in graph-friendly for loop."""
        batch_size = images.shape[0]
        original_dtype = images.dtype
        results = tf.TensorArray(
            dtype=original_dtype,
            size=batch_size,
            element_shape=tf.TensorShape(dims=images[0].shape),
        )
        for batch_id in tf.range(batch_size):
            single_tensor = tf.gather(images, batch_id)

            transformed_frame = self._equalize(single_tensor)
            transformed_frame = tf.cast(transformed_frame, original_dtype)

            results = results.write(batch_id, transformed_frame)

        return results.stack()

    # Todo: add source
    @staticmethod
    def _equalize(image):
        """Equalize function from PIL using TF ops."""

        def scale_channel(im, c):
            """Scale the data in the channel to implement equalize."""
            im = tf.cast(im[:, :, c], tf.int32)
            # Compute the histogram of the image channel.
            histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

            # For the purposes of computing the step, filter out the nonzeros.
            nonzero = tf.where(tf.not_equal(histo, 0))
            nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
            step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

            def build_lut(histo, step):
                # Compute the cumulative sum, shifting by step // 2
                # and then normalization by step.
                lut = (tf.cumsum(histo) + (step // 2)) // step
                # Shift lut, prepending with 0.
                lut = tf.concat([[0], lut[:-1]], 0)
                # Clip the counts to be in range.  This is done
                # in the C code for image.point.
                return tf.clip_by_value(lut, 0, 255)

            # If step is zero, return the original image.  Otherwise, build
            # lut from the full histogram and step and then index from it.
            result = tf.cond(
                tf.equal(step, 0),
                lambda: im,
                lambda: tf.gather(build_lut(histo, step), im),
            )

            return tf.cast(result, tf.uint8)

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(image, 0)
        s2 = scale_channel(image, 1)
        s3 = scale_channel(image, 2)
        image = tf.stack([s1, s2, s3], 2)
        return image

    def _parse_level(self, level: tf.Tensor) -> None:
        """Level unused by this op."""
        pass
