"""Contains code for AutoAugment operation."""
from typing import Callable

import tensorflow as tf

from tf_autoaugment.policies import get_transforms
from tf_autoaugment.transforms import NAME_TO_TRANSFORM


class AutoAugment:
    """Implements Autoaugment operation."""

    def __init__(self, policy_key: str):
        self.policy_key = policy_key

        self.transforms = get_transforms(policy_key)
        self.op_meta = self._get_op_meta()
        self.random_generator = tf.random.get_global_generator()

    def get_params(self):
        """Return randomly chosen: transform index, probabilities and signs."""
        num_transforms = len(self.transforms)
        transform_idx = self.random_generator.uniform(
            shape=(), maxval=num_transforms, dtype=tf.int32
        )
        probabilities = self.random_generator.normal(shape=(2,))
        signs = self.random_generator.uniform(shape=(2,), maxval=2, dtype=tf.int32)
        return transform_idx, probabilities, signs

    def __call__(self, images):
        """Run randomly chosen transforms pair over given input images."""
        transform_idx, probabilities, signs = self.get_params()
        for i, (op_name, p, magnitude_id) in enumerate(self.transforms[transform_idx]):
            if probabilities[i] <= p:
                magnitudes, signed = self.op_meta[op_name]
                magnitude = self.get_magnitude(magnitudes, magnitude_id)
                magnitude = self.invert_magnitude_based_on_sign(
                    magnitude=magnitude, signed=signed, sign=signs[i]
                )

                transform = self.get_transform(op_name)
                images = transform(images=images, magnitude=magnitude)

        return images

    @staticmethod
    def _get_op_meta():
        bins = 10
        return {
            "ShearX": (tf.linspace(0.0, 0.3, bins), True),
            "ShearY": (tf.linspace(0.0, 0.3, bins), True),
            "TranslateX": (tf.linspace(0.0, 150.0 / 331.0, bins), True),
            "TranslateY": (tf.linspace(0.0, 150.0 / 331.0, bins), True),
            "Rotate": (tf.linspace(0.0, 30.0, bins), True),
            "Brightness": (tf.linspace(0.0, 0.9, bins), True),
            "Color": (tf.linspace(0.0, 0.9, bins), True),
            "Contrast": (tf.linspace(0.0, 0.9, bins), True),
            "Sharpness": (tf.linspace(0.0, 0.9, bins), True),
            "Posterize": (tf.constant([8, 8, 7, 7, 6, 6, 5, 5, 4, 4]), False),
            "Solarize": (tf.linspace(256.0, 0.0, bins), False),
            "AutoContrast": (None, None),
            "Equalize": (None, None),
            "Invert": (None, None),
        }

    @staticmethod
    def get_magnitude(magnitudes, magnitude_id):
        """Return magnitude if provided, 0 otherwise."""
        return (
            magnitudes[magnitude_id]
            if magnitudes is not None or magnitude_id is not None
            else 0.0
        )

    @staticmethod
    def invert_magnitude_based_on_sign(magnitude, signed, sign):
        """Return inverted magnitude based on sign conditions."""
        if signed is not None and signed and sign == 0:
            magnitude *= -1.0
        return magnitude

    @staticmethod
    def get_transform(name: str) -> Callable:
        """Return transform operation from a set of available transforms."""
        return NAME_TO_TRANSFORM[name]
