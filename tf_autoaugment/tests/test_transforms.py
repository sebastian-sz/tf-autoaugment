import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized

from tf_autoaugment.transforms.autocontrast import AutoContrast
from tf_autoaugment.transforms.brightness import Brightness
from tf_autoaugment.transforms.color import Color
from tf_autoaugment.transforms.contrast import Contrast
from tf_autoaugment.transforms.cutout import CutOut
from tf_autoaugment.transforms.equalize import Equalize
from tf_autoaugment.transforms.invert import Invert
from tf_autoaugment.transforms.posterize import Posterize
from tf_autoaugment.transforms.rotate import Rotate
from tf_autoaugment.transforms.sharpness import Sharpness
from tf_autoaugment.transforms.shear_x import ShearX
from tf_autoaugment.transforms.shear_y import ShearY
from tf_autoaugment.transforms.solarize import Solarize
from tf_autoaugment.transforms.solarize_add import SolarizeAdd
from tf_autoaugment.transforms.translate_x import TranslateX
from tf_autoaugment.transforms.translate_y import TranslateY

ALL_TRANSFORMS = {
    "Invert": Invert,
    "Solarize": Solarize,
    "Equalize": Equalize,
    "Posterize": Posterize,
    "Rotate": Rotate,
    "AutoContrast": AutoContrast,
    "Brightness": Brightness,
    "Color": Color,
    "Contrast": Contrast,
    "Sharpness": Sharpness,
    "ShearX": ShearX,
    "ShearY": ShearY,
    "TranslateX": TranslateX,
    "TranslateY": TranslateY,
    "SolarizeAdd": SolarizeAdd,
    "CutOut": CutOut,
}
DTYPES = [tf.uint8, tf.int32, tf.float32]

TEST_NAMED_PARAMS = [
    {
        "testcase_name": f"{transform_name}_{dtype.name}",
        "transform_name": transform_name,
        "dtype": dtype,
    }
    for transform_name in list(ALL_TRANSFORMS.keys())
    for dtype in DTYPES
]


class TestTransformsUnit(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    level = rng.uniform(shape=(), minval=2, maxval=5, dtype=tf.int32)

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_transform_eager_mode(self, transform_name: str, dtype: tf.dtypes.DType):
        inputs = tf.cast(self.mock_input, dtype)
        transform = ALL_TRANSFORMS[transform_name]()

        outputs = transform(inputs, self.level)

        assert inputs.dtype == outputs.dtype
        assert inputs.shape == outputs.shape
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(outputs, inputs)

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_transform_graph_mode(self, transform_name: str, dtype: tf.dtypes.DType):
        inputs = tf.cast(self.mock_input, dtype)
        dataset = tf.data.Dataset.from_tensor_slices([inputs])

        transform = ALL_TRANSFORMS[transform_name]()
        dataset = dataset.map(lambda x: transform(x, self.level))
        outputs = tf.convert_to_tensor(list(dataset))[0]

        assert inputs.dtype == outputs.dtype
        assert inputs.shape == outputs.shape
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(outputs, inputs)

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_transform_tf_function(self, transform_name: str, dtype: tf.dtypes.DType):
        inputs = tf.cast(self.mock_input, dtype)
        transform = ALL_TRANSFORMS[transform_name](tf_function=True)

        outputs = transform(inputs, self.level)

        assert inputs.dtype == outputs.dtype
        assert inputs.shape == outputs.shape
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(outputs, inputs)

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_tf_function_is_not_getting_retraced(
        self, transform_name: str, dtype: tf.dtypes.DType
    ):
        inputs = tf.cast(self.mock_input, dtype)
        # Pass new arguments to test if fn will be retraced.
        new_inputs = self.rng.uniform(
            shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
        )
        new_inputs = tf.cast(new_inputs, dtype)
        new_level = self.level - 1

        transform = ALL_TRANSFORMS[transform_name](tf_function=True)

        f1 = transform.__call__.get_concrete_function(inputs, level=self.level)
        f2 = transform.__call__.get_concrete_function(new_inputs, level=new_level)
        assert isinstance(f1.graph, tf.Graph)
        assert isinstance(f2.graph, tf.Graph)

        # Assert the function has not been retraced:
        assert f1 is f2


class TestTransformsOutputConsistency(parameterized.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()
    mock_input = rng.uniform(
        shape=(5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )
    level = rng.uniform(shape=(), minval=1, maxval=5, dtype=tf.int32)

    # The transforms were originally designed for uint8
    mock_input = tf.cast(mock_input, tf.uint8)
    tolerance = 5  # pixel values for 0-255 range.

    @parameterized.named_parameters(TEST_NAMED_PARAMS)
    def test_output_is_consistent_between_dtypes(
        self, transform_name: str, dtype: tf.dtypes.DType
    ):
        transform = ALL_TRANSFORMS[transform_name]()
        expected_outputs = transform._debug_run(self.mock_input, self.level)  # uint8

        inputs = tf.cast(self.mock_input, dtype)
        outputs = transform._debug_run(inputs, self.level)
        outputs = tf.cast(outputs, tf.uint8)

        np.testing.assert_allclose(
            outputs, expected_outputs, rtol=self.tolerance, atol=self.tolerance
        )


if __name__ == "__main__":
    absltest.main()
