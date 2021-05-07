"""Unit tests for batched, modified versions of transforms.

We want to check if our modified, batched version work the same way as non-batched
executed in a for loop.
"""
import tensorflow as tf
import tensorflow_addons as tfa
from absl.testing import absltest, parameterized

from tf_autoaugment.transforms.autocontrast import AutoContrast
from tf_autoaugment.transforms.contrast import Contrast
from tf_autoaugment.transforms.equalize import Equalize
from tf_autoaugment.transforms.shear_x import ShearX
from tf_autoaugment.transforms.shear_y import ShearY
from tf_autoaugment.transforms.translate_x import TranslateX
from tf_autoaugment.transforms.translate_y import TranslateY

INPUT_DTYPES = [tf.float32, tf.int32, tf.uint8]
NAMED_TEST_PARAMS = [
    {"testcase_name": dtype.name, "dtype": dtype} for dtype in INPUT_DTYPES
]


class TestBatchedAutocontrast(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    inputs = tf.cast(mock_input, tf.uint8)  # autocontrast expects uint8

    transform = AutoContrast()

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_autocontrast_working_the_same_way_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        non_batched_output = tf.stack(
            [self.transform._autocontrast(x) for x in self.inputs]
        )

        # Batched is dtype agnostic:
        batched_input = tf.cast(self.mock_input, dtype)
        batched_output = self.transform.autocontrast(images=batched_input)

        tf.debugging.assert_equal(
            tf.cast(batched_output, tf.uint8),  # non-batched autocontrast returns uint8
            non_batched_output,
        )


class TestBatchedContrast(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    inputs = tf.cast(mock_input, tf.uint8)  # contrast expects uint8

    factor = rng.uniform(shape=(), minval=0, maxval=1.0, dtype=tf.float32)

    transform = Contrast()

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_contrast_working_the_same_way_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        non_batched_output = tf.stack(
            [self.transform._contrast(x, self.factor) for x in self.inputs]
        )

        # Batched is dtype agnostic:
        batched_input = tf.cast(self.inputs, dtype)
        batched_output = self.transform.contrast(batched_input, self.factor)

        tf.debugging.assert_equal(
            tf.cast(batched_output, tf.uint8),  # non-batched contrast return uint8
            non_batched_output,
        )


class TestBatchedEqualize(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    transform = Equalize()

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_equalize_working_the_same_way_as_the_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(self.mock_input, dtype)
        non_batched_output = tf.stack([self.transform._equalize(x) for x in inputs])

        batched_output = self.transform.equalize(inputs)

        tf.debugging.assert_equal(
            tf.cast(batched_output, tf.uint8),  # non-batched equalize returns uint8
            non_batched_output,
        )


class TestBatchedShearX(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    level = rng.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32)
    replace = tf.constant([128] * 3)

    transform = ShearX()

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_shear_x_producing_same_outputs_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(self.mock_input, dtype)
        non_batched_output = tf.stack(
            [
                tfa.image.shear_x(x, level=self.level, replace=self.replace)
                for x in inputs
            ]
        )
        batched_output = self.transform.shear_x(
            inputs, level=self.level, replace=self.replace
        )

        tf.debugging.assert_equal(batched_output, non_batched_output)


class TestBatchedShearY(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    level = rng.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32)
    replace = tf.constant([128] * 3)

    transform = ShearY()

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_shear_y_producing_same_outputs_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(self.mock_input, dtype)
        non_batched_output = tf.stack(
            [
                tfa.image.shear_y(x, level=self.level, replace=self.replace)
                for x in inputs
            ]
        )
        batched_output = self.transform.shear_y(
            inputs, level=self.level, replace=self.replace
        )

        tf.debugging.assert_equal(batched_output, non_batched_output)


class TestBatchedTranslateX(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    pixels = rng.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32)
    replace = tf.constant([128] * 3)

    transform = TranslateX()

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_translate_x_works_the_same_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(self.mock_input, dtype)

        non_batched_output = tf.stack(
            [
                tfa.image.translate_xy(
                    x, replace=self.replace, translate_to=[-1 * self.pixels, 0]
                )
                for x in inputs
            ]
        )

        batched_output = self.transform.translate_x(
            inputs, replace=self.replace, pixels=self.pixels
        )

        tf.debugging.assert_equal(batched_output, non_batched_output)


class TestBatchedTranslateY(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    pixels = rng.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32)
    replace = tf.constant([128] * 3)

    transform = TranslateY()

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_batched_translate_x_works_the_same_as_non_batched_in_for_loop(
        self, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(self.mock_input, dtype)

        non_batched_output = tf.stack(
            [
                tfa.image.translate_xy(
                    x, replace=self.replace, translate_to=[0, -1 * self.pixels]
                )
                for x in inputs
            ]
        )

        batched_output = self.transform.translate_y(
            inputs, replace=self.replace, pixels=self.pixels
        )

        tf.debugging.assert_equal(batched_output, non_batched_output)


if __name__ == "__main__":
    absltest.main()
