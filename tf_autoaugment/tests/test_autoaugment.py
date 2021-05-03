"""Contains tests for autoaugment class."""
import pytest
import tensorflow as tf

from tf_autoaugment.autoaugment import AutoAugment
from tf_autoaugment.transforms import NAME_TO_TRANSFORM


@pytest.fixture
def autoaugment():
    return AutoAugment(policy_key="imagenet")


def test_autoaugment_getting_right_params(autoaugment):
    params = autoaugment.get_params()

    assert isinstance(params, tuple)
    assert len(params) == 3
    assert isinstance(params[0], tf.Tensor)
    assert isinstance(params[1], tf.Tensor)
    assert isinstance(params[2], tf.Tensor)
    assert params[0].shape == ()
    assert params[1].shape == (2,)
    assert params[2].shape == (2,)


def test_autoaugment_applying_augmentation(autoaugment):
    original_frames = tf.random.uniform(
        (5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )  # Todo: add float compatibility.

    transformed_images = autoaugment(original_frames)

    assert transformed_images.shape == original_frames.shape


@pytest.mark.parametrize("transform_name", list(NAME_TO_TRANSFORM.keys()))
def test_transforms_are_created_only_once(transform_name, autoaugment):
    transform_1 = autoaugment.get_transform(transform_name)
    transform_2 = autoaugment.get_transform(transform_name)

    assert id(transform_1) == id(transform_2)
