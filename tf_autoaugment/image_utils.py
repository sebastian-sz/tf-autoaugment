import tensorflow as tf


def wrap_batch(image):
    """Return images batch with an extra channel set to all 1s.

    This is a modified version of the script present in the original repository:
    https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L404  # noqa: E501
    """
    shape = tf.shape(image)
    extended_channel = tf.ones([shape[0], shape[1], shape[2], 1], image.dtype)

    extended = tf.concat([image, extended_channel], -1)
    return extended


def unwrap_batch(image, replace):
    """Unwraps an images batch produced by wrap_batch.

    Where there is a 0 in the last channel for every spatial position,
    the rest of the three channels in that spatial dimension are grayed
    (set to 128).  Operations like translate and shear on a wrapped
    Tensor will leave 0s in empty locations. Some transformations look
    at the intensity of values to do preprocessing, and we want these
    empty pixels to assume the 'average' value, rather than pure black.

    Args:
        image: A 4D Images Tensor with 4 channels.
        replace: A one or three value 1D tensor to fill empty pixels.
    Returns:
        image: A 4D Images Tensor with 3 channels.

    This is a modified version of the script present in the original repository:
    https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L412 # noqa: E501
    """
    image_shape = tf.shape(image)
    # Flatten the spatial dimensions.
    flattened_image = tf.reshape(image, [-1, image_shape[3]])

    # Find all pixels where the last channel is zero.
    alpha_channel = flattened_image[:, 3]
    replace = tf.cast(replace, image.dtype)

    replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

    # Where they are zero, fill them in with 'replace'.
    flattened_image = tf.where(
        tf.expand_dims(tf.equal(alpha_channel, 0), axis=-1),
        tf.ones_like(flattened_image, dtype=image.dtype) * replace,
        flattened_image,
    )

    image = tf.reshape(flattened_image, image_shape)
    image = tf.slice(
        image, [0, 0, 0, 0], [image_shape[0], image_shape[1], image_shape[2], 3]
    )
    return image


def blend_batch(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
    """Blend image1 and image2 using 'factor'.

    Factor can be above 0.0.  A value of 0.0 means only image1 is used.
    A value of 1.0 means only image2 is used.  A value between 0.0 and
    1.0 means we linearly interpolate the pixel values between the two
    images.  A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.

    :param image1: images Tensor of shape (batch, height, width, channels)
    :param image2: images Tensor of shape (batch, height, width, channels)
    :param factor A floating point value above 0.0.
    :return tf.Tensor. A blended image.
    """
    assert image1.dtype == image2.dtype, "Both images need to be the same dtype."
    original_dtype = image1.dtype

    if factor == 0.0:
        return image1
    if factor == 1.0:
        return image2

    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    difference = image2 - image1
    scaled = factor * difference

    temp = image1 + scaled

    # Interpolate
    if factor > 0.0 and factor < 1.0:
        # Interpolation means we always stay within 0 and 255.
        result = tf.cast(temp, tf.uint8)
    else:
        # Extrapolate:
        # We need to clip and then cast.
        result = tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)

    return tf.cast(result, original_dtype)
