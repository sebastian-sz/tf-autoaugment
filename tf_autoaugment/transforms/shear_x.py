"""Code for ShearX Transform."""
from typing import List, Tuple, Union

import tensorflow as tf
import tensorflow_addons as tfa

from tf_autoaugment.image_utils import unwrap_batch, wrap_batch
from tf_autoaugment.transforms.base_transform import BaseTransform


class ShearX(BaseTransform):
    """Implements ShearX Transform."""

    def __init__(
        self,
        fill_value: Union[int, List, Tuple, tf.Tensor] = 128,
        tf_function: bool = False,
    ):
        super().__init__(tf_function=tf_function)

        self.fill_value = self._parse_fill_value(fill_value)

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run shear x function on image batch."""
        level = self._parse_level(level)
        return self.shear_x(images=images, level=level, replace=self.fill_value)

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        level = (level / self._MAX_LEVEL) * 0.3
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return level

    @staticmethod
    def shear_x(images: tf.Tensor, replace: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
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

    def _debug_run(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        level = (level / self._MAX_LEVEL) * 0.3
        return self.shear_x(images=images, level=level, replace=self.fill_value)
