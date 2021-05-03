"""
Contains tests regarding if batched version work properly.

In more detail, we wish to test whether batched operation works the same way
as non batched operation, executed in a for loop.
"""
import math

import pytest
import tensorflow as tf
import tensorflow_addons as tfa
from absl.testing import absltest

from tf_autoaugment.raw_img_ops import (
    autocontrast,
    batched_autocontrast,
    batched_equalize,
    batched_shear_x,
    batched_shear_y,
    equalize,
    translate_x,
)
from tf_autoaugment.utils import unwrap, wrap


@pytest.fixture
def mock_images():
    return tf.random.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )


def test_batched_autocontrast(mock_images):
    non_batched_output = tf.stack([autocontrast(x) for x in mock_images])

    batched_output = batched_autocontrast(mock_images)

    tf.debugging.assert_equal(non_batched_output, batched_output)


def test_batched_equalize(mock_images):
    non_batched_output = tf.stack([equalize(x) for x in mock_images])

    batched_output = batched_equalize(mock_images)

    tf.debugging.assert_equal(non_batched_output, batched_output)


def test_batched_shear_x(mock_images):
    fill_value = 128
    level = math.degrees(5)  # Random number tbh.

    non_batched_output = tf.stack(
        [tfa.image.shear_x(x, replace=fill_value, level=level) for x in mock_images]
    )

    batched_output = batched_shear_x(mock_images, replace=fill_value, level=level)

    tf.debugging.assert_equal(non_batched_output, batched_output)


def test_batched_shear_y(mock_images):
    fill_value = 128
    level = math.degrees(5)  # Random number tbh.

    non_batched_output = tf.stack(
        [tfa.image.shear_y(x, replace=fill_value, level=level) for x in mock_images]
    )

    batched_output = batched_shear_y(mock_images, replace=fill_value, level=level)

    tf.debugging.assert_equal(non_batched_output, batched_output)


class TestBatchedTranslateX(absltest.TestCase):
    noise = tf.random.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    fill_value = 128
    magnitude = 5  # Todo: random
    pixels = noise.shape[1] * magnitude

    @staticmethod
    def _translate_x(image, pixels, replace):
        """Equivalent of PIL Translate in X dimension."""
        image = tfa.image.translate(wrap(image), [-pixels, 0])
        return unwrap(image, replace)

    def test_batched_and_non_batched_produce_the_same_outputs(self):
        non_batched_output = tf.stack(
            [
                self._translate_x(x, replace=[self.fill_value] * 3, pixels=self.pixels)
                for x in self.noise
            ]
        )

        batched_output = translate_x(
            self.noise, replace=self.fill_value, pixels=self.pixels
        )

        tf.debugging.assert_equal(non_batched_output, batched_output)


class TestBatchedTranslateY(absltest.TestCase):
    noise = tf.random.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    fill_value = 128
    magnitude = 5  # Todo: random
    pixels = noise.shape[2] * magnitude

    @staticmethod
    def _translate_y(image, pixels, replace):
        """Equivalent of PIL Translate in X dimension."""
        image = tfa.image.translate(wrap(image), [0, -pixels])
        return unwrap(image, replace)

    def test_batched_and_non_batched_produce_the_same_outputs(self):
        non_batched_output = tf.stack(
            [
                self._translate_y(x, replace=[self.fill_value] * 3, pixels=self.pixels)
                for x in self.noise
            ]
        )

        batched_output = translate_x(
            self.noise, replace=self.fill_value, pixels=self.pixels
        )

        tf.debugging.assert_equal(non_batched_output, batched_output)
