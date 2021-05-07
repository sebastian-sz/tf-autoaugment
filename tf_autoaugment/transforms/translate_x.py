"""Code for Translate X Transform."""

from typing import List, Tuple, Union

import tensorflow as tf
import tensorflow_addons as tfa

from tf_autoaugment.image_utils import unwrap_batch, wrap_batch
from tf_autoaugment.transforms.base_transform import BaseTransform


class TranslateX(BaseTransform):
    """Implements Translate X Transform."""

    def __init__(
        self,
        fill_value: Union[int, List, Tuple, tf.Tensor] = 128,
        translate_const: int = 250,
        tf_function: bool = False,
    ):
        super().__init__(tf_function=tf_function)

        self.fill_value = self._parse_fill_value(fill_value)
        self.translate_const = translate_const

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run translate_x function on image batch."""
        level = self._parse_level(level)
        return self.translate_x(images=images, replace=self.fill_value, pixels=level)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        level = (level / self._MAX_LEVEL) * self.translate_const
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return level

    @staticmethod
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

    def _debug_run(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        level = (level / self._MAX_LEVEL) * float(self.translate_const)
        return self.translate_x(images=images, replace=self.fill_value, pixels=level)
