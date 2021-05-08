# type: ignore
"""Code for Base Transform."""
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import tensorflow as tf


class BaseTransform(ABC):
    """Base class for Images Transforms."""

    _MAX_LEVEL = tf.constant(10, dtype=tf.int32)

    def __init__(self, tf_function: bool = False):
        self.__call__ = (
            tf.function(func=self.__call__, experimental_compile=True)
            if tf_function
            else self.__call__
        )

    @abstractmethod
    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run transform on images batch."""
        raise NotImplementedError

    @abstractmethod
    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        """Parse level value to be used for transform function."""
        raise NotImplementedError

    @staticmethod
    def _randomly_negate_tensor(tensor: tf.Tensor) -> tf.Tensor:
        """With 50% probability turn the tensor negative."""
        should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
        final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
        return final_tensor

    def _debug_run(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Apply transform without the randomness.

        Some transforms make use of self._randomly_negate_tensor which makes it
        impossible to test consistency between invocations. This is a wrapper method
        which is meant to run the transform in the same way each time the transform
        is invoked.

        Defaults to self.__call__ method.
        """
        return self.__call__(images=images, level=level)

    @staticmethod
    def _parse_fill_value(fill_value: Union[int, List, Tuple, tf.Tensor]) -> tf.Tensor:
        """
        Create a 1-D, 3 element tensor out of provided value.

        Only used in Shear and Translate OPs.
        """
        if isinstance(fill_value, int):
            return tf.constant([fill_value] * 3, dtype=tf.int32)
        elif isinstance(fill_value, list) or isinstance(fill_value, tuple):
            assert len(fill_value) == 3, "fill_value must be int or 3 element sequence"
            return tf.constant(fill_value, dtype=tf.int32)
        elif isinstance(fill_value, tf.Tensor):
            assert fill_value.shape == 3, "fill_value must be int or 3 element sequence"
            return fill_value
