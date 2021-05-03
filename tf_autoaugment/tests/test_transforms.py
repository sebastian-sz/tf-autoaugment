"""Contains test to sanity check whether each transform can be executed."""

from random import randint

import numpy as np
import pytest
import tensorflow as tf

from tf_autoaugment.transforms import NAME_TO_TRANSFORM


@pytest.mark.parametrize("transform_name", list(NAME_TO_TRANSFORM.keys()))
def test_eager_transforms(transform_name):
    magnitude = randint(1, 7)
    original_frames = tf.random.uniform(
        (5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )  # Todo: add float compatibility.

    transform = NAME_TO_TRANSFORM[transform_name]
    transformed_images = transform(original_frames, magnitude=magnitude)

    assert transformed_images.shape == original_frames.shape
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(transformed_images, original_frames)


@pytest.mark.parametrize("transform_name", list(NAME_TO_TRANSFORM.keys()))
def test_graph_transforms(transform_name):
    magnitude = randint(1, 7)
    original_frames = tf.random.uniform(
        (5, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
    )  # Todo: add float compatibility.
    dataset = tf.data.Dataset.from_tensor_slices([original_frames])

    transform = NAME_TO_TRANSFORM[transform_name]
    dataset = dataset.map(lambda x: transform(x, magnitude=magnitude))

    transformed_frames = np.array(list(dataset))[0]

    assert transformed_frames.shape == original_frames.shape
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(transformed_frames, original_frames)
