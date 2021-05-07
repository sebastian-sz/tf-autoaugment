"""Contains code for CutOut Transform."""
from typing import List, Tuple, Union

import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform


class CutOut(BaseTransform):
    """Implements CutOut Transform."""

    def __init__(
        self,
        fill_value: Union[int, List, Tuple, tf.Tensor] = 128,
        cutout_const: int = 250,
        tf_function: bool = False,
    ):
        super().__init__(tf_function=tf_function)

        self.fill_value = self._parse_fill_value(fill_value)
        self.cutout_const = cutout_const

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run cutout on image batch."""
        pad_size = self._parse_level(level)
        return self.cutout(
            image=images, pad_size=pad_size, replace=self.fill_value
        )  # Todo: make sure this is ok.

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        result = (level / self._MAX_LEVEL) * 110
        return tf.cast(result, tf.int32)

    @staticmethod
    def cutout(
        image: tf.Tensor, pad_size: tf.Tensor, replace: tf.Tensor = 0
    ) -> tf.Tensor:
        """
        Apply cutout (https://arxiv.org/abs/1708.04552) to image.

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

    def _debug_run(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        pad_size = self._parse_level(level)
        return self._debug_cutout(images, pad_size, self.fill_value)

    @staticmethod
    def _debug_cutout(image: tf.Tensor, pad_size: tf.Tensor, replace: tf.Tensor = 0):
        """Only for testing. Cutout always the same part of the image."""
        image_height = tf.shape(image)[1]
        image_width = tf.shape(image)[2]

        replace = tf.cast(replace, image.dtype)

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = 10
        cutout_center_width = 10

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
