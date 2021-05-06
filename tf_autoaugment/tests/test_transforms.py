import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from tf_autoaugment.transforms.transforms import NAME_TO_TRANSFORM

DTYPES_TO_TEST = [tf.int32, tf.uint8, tf.float32]
TRANSFORMS_TO_TEST = list(NAME_TO_TRANSFORM.keys())

TEST_NAMED_PARAMS = [  # Named params for pretty printing
    {
        "testcase_name": f"{transform_name}_{dtype.name}",
        "transform_name": transform_name,
        "dtype": dtype,
    }
    for transform_name in TRANSFORMS_TO_TEST
    for dtype in DTYPES_TO_TEST
]


class TestTransformsForwardPass(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    level = int(rng.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32))

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_transform_eager_mode(self, transform_name: str, dtype: tf.dtypes.DType):
        inputs = tf.cast(self.mock_input, dtype)
        transform = NAME_TO_TRANSFORM[transform_name]

        outputs = transform(inputs, self.level)

        assert inputs.dtype == outputs.dtype
        assert inputs.shape == outputs.shape
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(outputs, inputs)

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_transform_graph_mode(self, transform_name: str, dtype: tf.dtypes.DType):
        inputs = tf.cast(self.mock_input, dtype)
        dataset = tf.data.Dataset.from_tensor_slices([inputs])

        transform = NAME_TO_TRANSFORM[transform_name]
        dataset = dataset.map(lambda x: transform(x, self.level))
        outputs = tf.convert_to_tensor(list(dataset))[0]

        assert inputs.dtype == outputs.dtype
        assert inputs.shape == outputs.shape
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(outputs, inputs)

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_transform_tf_function(self, transform_name: str, dtype: tf.dtypes.DType):
        inputs = tf.cast(self.mock_input, dtype)
        transform = NAME_TO_TRANSFORM[transform_name]
        transform.__call__ = tf.function(transform.__call__)

        outputs = transform(inputs, self.level)

        assert inputs.dtype == outputs.dtype
        assert inputs.shape == outputs.shape
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(outputs, inputs)


class TestTransformsOutputConsistency(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    level = int(rng.uniform(shape=(), minval=1, maxval=7, dtype=tf.int32))

    # The transforms were originally designed for uint8
    mock_input = tf.cast(mock_input, tf.uint8)
    tolerance = 5  # pixel values for 0-255 range.

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_output_is_consistent_between_dtypes(
        self, transform_name: str, dtype: tf.dtypes.DType
    ):
        transform = NAME_TO_TRANSFORM[transform_name]
        expected_outputs = transform._debug_apply(self.mock_input, self.level)  # uint8

        inputs = tf.cast(self.mock_input, dtype)
        outputs = transform._debug_apply(inputs, self.level)
        outputs = tf.cast(outputs, tf.uint8)

        np.testing.assert_allclose(
            outputs, expected_outputs, rtol=self.tolerance, atol=self.tolerance
        )


if __name__ == "__main__":
    absltest.main()
