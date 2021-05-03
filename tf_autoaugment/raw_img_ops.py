"""
Contains all transform types present in AutoAugment.

Most of these operations are translated to TF 2.x from:
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py  # noqa: E501
"""

# Todo: add float32 compatibility.

import tensorflow as tf
import tensorflow_addons as tfa

from tf_autoaugment.utils import unwrap_batch, wrap_batch


def posterize(images: tf.Tensor, bits: int) -> tf.Tensor:
    """
    Equivalent of PIL Posterize, rewritten in Tensorflow.

    :param images: input images. Tensor of shape (batch, height, width, channels).
    :param bits: integer.
    :return: transformed images. Tensor of shape (batch, height, width, channels).

    Source:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L222  # noqa: E501
    """
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(images, shift), shift)


def solarize(images: tf.Tensor, threshold: int = 128):
    """
    Solarize transform.

    For each pixel in the image, select the pixel if the value is less than the
    threshold. Otherwise, subtract 255 from the pixel.

    :param images: input images.
    :param threshold: integer.
    :return: transformed image.

    Source:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L176 # noqa: E501
    """
    return tf.where(images < threshold, images, 255 - images)


def autocontrast(image):
    """
    Autocontrast function from PIL using TF ops.

    :param image: A 3D uint8 tensor.
    :return The image after it has had autocontrast applied to it and will be of type
      int32.
    """

    def scale_channel(img):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire
        # image to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(img), tf.float32)
        hi = tf.cast(tf.reduce_max(img), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.int32)

        # TODO: Consider vectorising this OP with tf.where
        result = tf.cond(hi > lo, lambda: scale_values(img), lambda: img)
        return result

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image[:, :, 0])
    s2 = scale_channel(image[:, :, 1])
    s3 = scale_channel(image[:, :, 2])
    image = tf.stack([s1, s2, s3], 2)
    return image


def batched_autocontrast(images: tf.Tensor) -> tf.Tensor:
    """
    Autocontrast wrapped in a graph-friendly for loop.

    :param images: tensor of shape (batch, height, width, channels).
    :return transformed image batch. tensor of shape (batch, height, width, channels).
    """
    batch_size = images.shape[0]

    results = tf.TensorArray(tf.int32, size=batch_size)

    for batch_id in tf.range(batch_size):
        single_tensor = tf.gather(images, batch_id)
        transformed_frame = autocontrast(single_tensor)
        results = results.write(batch_id, transformed_frame)

    return results.stack()


def invert(images: tf.Tensor) -> tf.Tensor:
    """Inverts the image pixels.

    :param images: tf.Tensor.
    :return: An inverted image Tensor of the same shape as input.
    """
    return 255 - images


def equalize(image):
    """
    Equalize function from PIL using TF ops.

    :param image: tf.Tensor of shape (height, width, channels).
    :return transformed image. tf.Tensor of shape (height, width, channels).
    """

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
            tf.equal(step, 0), lambda: im, lambda: tf.gather(build_lut(histo, step), im)
        )

        return tf.cast(result, tf.int32)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


def batched_equalize(images):
    """
    Equalize wrapped in a graph-friendly for loop.

    :param images. tf.Tensor of shape (batch, height, width, channels).
    :return transformed images. tf.Tensor of shape (batch, height, width, channels).
    """
    batch_size = images.shape[0]

    results = tf.TensorArray(tf.int32, size=batch_size)

    for batch_id in tf.range(batch_size):
        single_tensor = tf.gather(images, batch_id)
        transformed_frame = equalize(single_tensor)
        results = results.write(batch_id, transformed_frame)

    return results.stack()


def batched_shear_x(images: tf.Tensor, replace: int, level) -> tf.Tensor:
    """
    Batched version of shear_x function.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param replace. Value to replace the pixels with.
    :param level:. # Todo: docstring
    :return transformed images. tf.Tensor of shape (batch, height, width, channels)

    This is a modified version of the shear_x function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L265  # noqa: E501
    """
    # Todo: move this outside. Do not repeat for each invocation.
    if isinstance(replace, int):  # tile 3 times.
        replace = tf.constant([replace] * 3)

    images = tfa.image.transform(
        wrap_batch(images), [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    return unwrap_batch(images, replace)


def batched_shear_y(images: tf.Tensor, replace: int, level) -> tf.Tensor:
    """
    Batched version of shear_y function.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param replace. Value to replace the pixels with.
    :param level:. # Todo: docstring
    :return transformed images. tf.Tensor of shape (batch, height, width, channels)

    This is a modified version of the shear_y function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L276  # noqa: E501

    """
    # Todo: move this outside. Do not repeat for each invocation.
    if isinstance(replace, int):  # tile 3 times.
        replace = tf.constant([replace] * 3)

    images = tfa.image.transform(
        wrap_batch(images), [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0]
    )

    return unwrap_batch(images, replace)


def translate_x(images: tf.Tensor, pixels, replace: int) -> tf.Tensor:
    """
    Batched version of Translate in X dimension.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param pixels:. # Todo:
    :param replace. Value to replace the pixels with.

    This is a modified version of translate_x function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L253  # noqa: E501
    """
    # Todo: move this outside. Do not repeat for each invocation.
    if isinstance(replace, int):  # tile 3 times.
        replace = tf.constant([replace] * 3)

    images = tfa.image.translate(wrap_batch(images), [-pixels, 0])
    return unwrap_batch(images, replace)


# Todo: double check what should be the pixels value in AutoAugment.
def translate_y(images: tf.Tensor, pixels, replace: int) -> tf.Tensor:
    """
    Batched version of Translate in Y dimension.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param pixels:. # Todo:
    :param replace. Value to replace the pixels with.

    This is a modified version of translate_x function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L259  # noqa: E501

    """
    # Todo: move this outside. Do not repeat for each invocation.
    if isinstance(replace, int):  # tile 3 times.
        replace = tf.constant([replace] * 3)

    images = tfa.image.translate(wrap_batch(images), [0, -pixels])
    return unwrap_batch(images, replace)
