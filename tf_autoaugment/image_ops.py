"""Image operations

Most of these are rewritten using TF 2.x, Tensorflow Addons, and modified to support
batched input, rather than single image tensor.
Source:
    https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py
"""
import tensorflow as tf
import tensorflow_addons as tfa

from tf_autoaugment.image_utils import blend_batch, unwrap_batch, wrap_batch


def solarize(images: tf.Tensor, threshold=128):
    """Invert all pixels above a threshold value."""
    return tf.where(images < threshold, images, 255 - images)


def solarize_add(images, addition=0, threshold=128):
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


def color(images: tf.Tensor, factor):
    """Adjust the color balance of the image."""
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(images))
    return blend_batch(degenerate, images, factor)


def brightness(images, factor):
    """Equivalent of PIL Brightness."""
    degenerate = tf.zeros_like(images)
    return blend_batch(degenerate, images, factor)


def posterize(images, bits):
    """Reduce the number of bits for each pixel."""
    assert images.dtype == tf.uint8
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(images, shift), shift)


def invert(images):
    """Invert the pixels of the image."""
    return 255 - images


def _contrast(image, factor):
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


def contrast(images, factor):
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

        transformed_frame = _contrast(single_tensor, factor)
        transformed_frame = tf.cast(transformed_frame, original_dtype)

        results = results.write(batch_id, transformed_frame)

    return results.stack()


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


def autocontrast(images: tf.Tensor) -> tf.Tensor:
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

        transformed_frame = _autocontrast(single_tensor)
        transformed_frame = tf.cast(transformed_frame, original_dtype)

        results = results.write(batch_id, transformed_frame)

    return results.stack()


def sharpness(images, factor):
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

    # For the borders of the resulting image, fill in the values of the original image.
    mask = tf.ones_like(degenerate)
    padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
    padded_degenerate = tf.pad(degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])
    result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

    # Blend the final result.
    blended = blend_batch(result, orig_image, factor)
    return tf.cast(blended, image_dtype)


def _equalize(image):
    """Implements Equalize function from PIL using TF ops."""

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

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image


def equalize(images):
    batch_size = images.shape[0]
    original_dtype = images.dtype
    results = tf.TensorArray(
        dtype=original_dtype,
        size=batch_size,
        element_shape=tf.TensorShape(dims=images[0].shape),
    )
    for batch_id in tf.range(batch_size):
        single_tensor = tf.gather(images, batch_id)
        # single_tensor = tf.cast(single_tensor, tf.uint8)

        transformed_frame = _equalize(single_tensor)
        transformed_frame = tf.cast(transformed_frame, original_dtype)

        results = results.write(batch_id, transformed_frame)

    return results.stack()


def shear_x(images: tf.Tensor, replace: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
    # Todo: test this batched vs non-batched.
    """
    Batched version of shear_x function.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param replace. Value to replace the pixels with.
    :param level:. # Todo: docstring
    :return transformed images. tf.Tensor of shape (batch, height, width, channels)

    This is a modified version of the shear_x function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L265  # noqa: E501
    """
    images = tfa.image.transform(
        wrap_batch(images), [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    return unwrap_batch(images, replace)


def shear_y(images: tf.Tensor, replace: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
    # Todo: test this batched vs non-batched.
    """
    Batched version of shear_y function.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param replace. Value to replace the pixels with.
    :param level:. # Todo: docstring
    :return transformed images. tf.Tensor of shape (batch, height, width, channels)

    This is a modified version of the shear_y function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L276  # noqa: E501

    """
    images = tfa.image.transform(
        wrap_batch(images), [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0]
    )

    return unwrap_batch(images, replace)


def translate_x(images: tf.Tensor, pixels, replace: tf.Tensor) -> tf.Tensor:
    """
    Batched version of Translate in X dimension.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param pixels:. # Todo:
    :param replace. Value to replace the pixels with.

    This is a modified version of translate_x function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L253  # noqa: E501
    """
    images = tfa.image.translate(wrap_batch(images), [-pixels, 0])
    return unwrap_batch(images, replace)


def translate_y(images: tf.Tensor, pixels, replace: tf.Tensor) -> tf.Tensor:
    """
    Batched version of Translate in Y dimension.

    :param images. tf.Tensor of shape (batch, height, width, channels)
    :param pixels:. # Todo:
    :param replace. Value to replace the pixels with.

    This is a modified version of translate_x function from:
        https://github.com/tensorflow/tpu/blob/0b33da3bdfc3706666dfa25631e0eed76a111e07/models/official/efficientnet/autoaugment.py#L259  # noqa: E501

    """
    images = tfa.image.translate(wrap_batch(images), [0, -pixels])
    return unwrap_batch(images, replace)


# Modified
def cutout(image, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.
    Args:
      image: An image Tensor of type uint8.
      pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
      replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.
    Returns:
      An image Tensor that is of type uint8.
    """
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]

    replace = tf.cast(replace, image.dtype)

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height, dtype=tf.int32
    )

    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=image_width, dtype=tf.int32
    )

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(
        tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image
    )
    return image
