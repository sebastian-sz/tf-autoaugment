import tensorflow as tf
from absl.testing import absltest, parameterized

from tf_autoaugment.image_utils import blend_batch, unwrap_batch, wrap_batch

INPUT_DTYPES = [tf.int32, tf.uint8, tf.float32]

NAMED_TEST_PARAMS = [
    {"testcase_name": dtype.name, "dtype": dtype} for dtype in INPUT_DTYPES
]


class TestBatchedWrap(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_images = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    @staticmethod
    def wrap(image):
        """
        Return 'image' with an extra channel set to all 1s.

        Source:
        https://github.com/tensorflow/tu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L404
        """
        shape = tf.shape(image)
        extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
        extended = tf.concat([image, extended_channel], 2)
        return extended

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batch_wrapped_producing_same_results_are_wrap_with_for_loop(self, dtype):
        inputs = tf.cast(self.mock_images, dtype)
        non_batched_output = tf.stack([self.wrap(x) for x in inputs])
        batched_output = wrap_batch(inputs)

        tf.debugging.assert_equal(non_batched_output, batched_output)


class TestBatchedUnwrap(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_images = rng.uniform(
        shape=(5, 224, 224, 4), minval=0, maxval=255, dtype=tf.int32
    )
    replace = tf.constant([128] * 3)

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batch_unwrap_producing_same_results_as_unwrap_with_for_loop(self, dtype):
        inputs = tf.cast(self.mock_images, dtype)
        non_batched_output = tf.stack([self.unwrap(x, self.replace) for x in inputs])
        batched_output = unwrap_batch(inputs, replace=self.replace)

        tf.debugging.assert_equal(non_batched_output, batched_output)

    @staticmethod
    def unwrap(image, replace):
        """
        Unwraps an image produced by wrap.

        Where there is a 0 in the last channel for every spatial position,
        the rest of the three channels in that spatial dimension are grayed
        (set to 128).  Operations like translate and shear on a wrapped
        Tensor will leave 0s in empty locations.  Some transformations look
        at the intensity of values to do preprocessing, and we want these
        empty pixels to assume the 'average' value, rather than pure black.
        Args:
          image: A 3D Image Tensor with 4 channels.
          replace: A one or three value 1D tensor to fill empty pixels.
        Returns:
          image: A 3D image Tensor with 3 channels.

        Source:
            https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L412 # noqa: E501
        """
        image_shape = tf.shape(image)
        # Flatten the spatial dimensions.
        flattened_image = tf.reshape(image, [-1, image_shape[2]])

        # Find all pixels where the last channel is zero.
        alpha_channel = flattened_image[:, 3]

        replace = tf.cast(replace, image.dtype)
        replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

        # Where they are zero, fill them in with 'replace'.
        flattened_image = tf.where(
            tf.expand_dims(tf.equal(alpha_channel, 0), axis=-1),
            # This had to be changed.
            tf.ones_like(flattened_image, dtype=image.dtype) * replace,
            flattened_image,
        )

        image = tf.reshape(flattened_image, image_shape)
        image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
        return image


class TestBatchedBlend(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_images = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    other_mock_images = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    factor = float(rng.uniform(shape=(), minval=0, maxval=1, dtype=tf.float32))

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_blend_producing_same_results_as_blend_with_for_loop(self, dtype):
        inputs = tf.cast(self.mock_images, dtype)
        other_inputs = tf.cast(self.other_mock_images, dtype)

        non_batched_output = tf.stack(
            [self.blend(x, y, self.factor) for x, y in zip(inputs, other_inputs)]
        )
        non_batched_output = tf.cast(non_batched_output, dtype)  # blend returns uint8

        batched_output = blend_batch(inputs, other_inputs, self.factor)

        tf.debugging.assert_equal(non_batched_output, batched_output)

    @staticmethod
    def blend(image1, image2, factor):
        """Blend image1 and image2 using 'factor'.
        Factor can be above 0.0.  A value of 0.0 means only image1 is used.
        A value of 1.0 means only image2 is used.  A value between 0.0 and
        1.0 means we linearly interpolate the pixel values between the two
        images.  A value greater than 1.0 "extrapolates" the difference
        between the two pixel values, and we clip the results to values
        between 0 and 255.
        Args:
            image1: An image Tensor of type uint8.
            image2: An image Tensor of type uint8.
            factor: A floating point value above 0.0.
        Returns:
            A blended image Tensor of type uint8.

        Source:
            https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L82  # noqa: E501
        """
        if factor == 0.0:
            return tf.convert_to_tensor(image1)
        if factor == 1.0:
            return tf.convert_to_tensor(image2)

        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)

        difference = image2 - image1
        scaled = factor * difference

        # Do addition in float.
        temp = tf.cast(image1, tf.float32) + scaled

        # Interpolate
        if 0.0 < factor < 1.0:
            # Interpolation means we always stay within 0 and 255.
            return tf.cast(temp, tf.uint8)

        # Extrapolate:
        #
        # We need to clip and then cast.
        return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


if __name__ == "__main__":
    absltest.main()
