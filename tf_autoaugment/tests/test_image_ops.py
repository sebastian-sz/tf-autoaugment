"""Unit tests for modified versions of image ops.

We want to check if our modified, batched version work the same way as non-batched
executed in a for loop.
"""

import tensorflow as tf
import tensorflow_addons as tfa
from absl.testing import absltest, parameterized

from tf_autoaugment.image_ops import (
    _autocontrast,
    _equalize,
    autocontrast,
    equalize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
)

INPUT_DTYPES = [tf.float32, tf.int32, tf.uint8]
NAMED_TEST_PARAMS = [
    {"testcase_name": dtype.name, "dtype": dtype} for dtype in INPUT_DTYPES
]

RNG = tf.random.Generator.from_non_deterministic_state()
MOCK_INPUT = RNG.uniform(shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32)


class TestBatchedAutocontrast(parameterized.TestCase):
    # Autocontrast expects uint8
    autocontrast_input = tf.cast(MOCK_INPUT, tf.uint8)

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_autocontrast_working_the_same_way_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        non_batched_output = tf.stack(
            [_autocontrast(x) for x in self.autocontrast_input]
        )

        # Batched is dtype agnostic:
        batched_input = tf.cast(MOCK_INPUT, dtype)
        batched_output = autocontrast(images=batched_input)

        tf.debugging.assert_equal(
            tf.cast(batched_output, tf.uint8),  # non-batched autocontrast returns uint8
            non_batched_output,
        )


class TestBatchedEqualize(parameterized.TestCase):
    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_equalize_working_the_same_way_as_the_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(MOCK_INPUT, dtype)
        non_batched_output = tf.stack([_equalize(x) for x in inputs])

        batched_output = equalize(inputs)

        tf.debugging.assert_equal(
            tf.cast(batched_output, tf.uint8),  # non-batched equalize returns uint8
            non_batched_output,
        )


class TestBatchedShearX(parameterized.TestCase):
    level = tf.constant(5)
    replace = tf.constant([128] * 3)

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_shear_x_producing_same_outputs_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(MOCK_INPUT, dtype)
        non_batched_output = tf.stack(
            [
                tfa.image.shear_x(x, level=self.level, replace=self.replace)
                for x in inputs
            ]
        )
        batched_output = shear_x(inputs, level=self.level, replace=self.replace)

        tf.debugging.assert_equal(batched_output, non_batched_output)


class TestBatchedShearY(parameterized.TestCase):
    level = tf.constant(5)
    replace = tf.constant([128] * 3)

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_shear_y_producing_same_outputs_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(MOCK_INPUT, dtype)
        non_batched_output = tf.stack(
            [
                tfa.image.shear_y(x, level=self.level, replace=self.replace)
                for x in inputs
            ]
        )
        batched_output = shear_y(inputs, level=self.level, replace=self.replace)

        tf.debugging.assert_equal(batched_output, non_batched_output)


class TestBatchedTranslateX(parameterized.TestCase):
    replace = tf.constant([128] * 3)
    pixels = tf.constant(20)  # Todo: random number

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_translate_x_works_the_same_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(MOCK_INPUT, dtype)

        non_batched_output = tf.stack(
            [
                tfa.image.translate_xy(
                    x, replace=self.replace, translate_to=[-1 * self.pixels, 0]
                )
                for x in inputs
            ]
        )

        batched_output = translate_x(inputs, replace=self.replace, pixels=self.pixels)

        tf.debugging.assert_equal(batched_output, non_batched_output)


class TestBatchedTranslateY(parameterized.TestCase):
    replace = tf.constant([128] * 3)
    pixels = tf.constant(20)  # Todo: random number

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_translate_y_works_the_same_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(MOCK_INPUT, dtype)

        non_batched_output = tf.stack(
            [
                tfa.image.translate_xy(
                    x, replace=self.replace, translate_to=[0, -1 * self.pixels]
                )
                for x in inputs
            ]
        )

        batched_output = translate_y(inputs, replace=self.replace, pixels=self.pixels)

        tf.debugging.assert_equal(batched_output, non_batched_output)


if __name__ == "__main__":
    absltest.main()
