from abc import ABC, abstractmethod
from typing import Any

import tensorflow as tf


class BaseTransform(ABC):
    """Base class for Images Transforms."""

    @abstractmethod
    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Run transform on images batch."""
        raise NotImplementedError

    @abstractmethod
    def _parse_level(self, level: int) -> Any:
        """Parse level value to be used for transform function."""
        raise NotImplementedError

    @staticmethod
    def _randomly_negate_tensor(tensor: tf.Tensor) -> tf.Tensor:
        """With 50% prob turn the tensor negative."""
        should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
        final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
        return final_tensor

    def _debug_apply(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Apply transform without the randomness.

        Some transforms make use of self._randomly_negate_tensor which makes it
        impossible to test consistency between invocations. This is a wrapper method
        which is meant to run the transform in the same way each time the transform
        is invoked.

        Defaults to self.__call__ method.
        """
        return self.__call__(images=images, level=level)
