import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from tf_autoaugment import RandAugment

NAMED_TEST_PARAMS = [
    {"testcase_name": "use_tf_function", "tf_function": True},
    {"testcase_name": "no_use_tf_function", "tf_function": False},
]


class TestRandAugment(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_inputs = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_randaugment_working_in_eager_mode(self, tf_function: bool):
        randaugment = RandAugment(tf_function=tf_function)

        transformed_images = randaugment(self.mock_inputs)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(self.mock_inputs, transformed_images)

    @parameterized.named_parameters(NAMED_TEST_PARAMS)
    def test_randaugment_working_in_graph_mode(self, tf_function: bool):
        randaugment = RandAugment(tf_function=tf_function)

        dataset = tf.data.Dataset.from_tensor_slices([self.mock_inputs])
        dataset = dataset.map(lambda x: randaugment(x))

        outputs = tf.convert_to_tensor(list(dataset))[0]
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(self.mock_inputs, outputs)


if __name__ == "__main__":
    absltest.main()
