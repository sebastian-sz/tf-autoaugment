# type: ignore
"""Code for AutoAugment operation."""
import tensorflow as tf
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer

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

POLICIES = {
    "_debug": [(("TranslateX", 1.0, 4), ("Equalize", 1.0, 10))],
    "imagenet": [
        (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)),
        (("Equalize", 0.8, 8), ("Equalize", 0.6, 3)),
        (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
        (("Equalize", 0.4, 7), ("Solarize", 0.2, 4)),
        (("Equalize", 0.4, 4), ("Rotate", 0.8, 8)),
        (("Solarize", 0.6, 3), ("Equalize", 0.6, 7)),
        (("Posterize", 0.8, 5), ("Equalize", 1.0, 2)),
        (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
        (("Equalize", 0.6, 8), ("Posterize", 0.4, 6)),
        (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
        (("Rotate", 0.4, 9), ("Equalize", 0.6, 2)),
        (("Equalize", 0.0, 7), ("Equalize", 0.8, 8)),
        (("Invert", 0.6, 4), ("Equalize", 1.0, 8)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
        (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
        (("Sharpness", 0.4, 7), ("Invert", 0.6, 8)),
        (("ShearX", 0.6, 5), ("Equalize", 1.0, 9)),
        (("Color", 0.4, 0), ("Equalize", 0.6, 3)),
        (("Equalize", 0.4, 7), ("Solarize", 0.2, 4)),
        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)),
        (("Invert", 0.6, 4), ("Equalize", 1.0, 8)),
        (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
        (("Equalize", 0.8, 8), ("Equalize", 0.6, 3)),
    ],
    "cifar10": [
        (("Invert", 0.1, 7), ("Contrast", 0.2, 6)),
        (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
        (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
        (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.5, 8), ("Equalize", 0.9, 2)),
        (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
        (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
        (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
        (("Equalize", 0.6, 5), ("Equalize", 0.5, 1)),
        (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
        (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
        (("Equalize", 0.3, 7), ("AutoContrast", 0.4, 8)),
        (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
        (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
        (("Solarize", 0.5, 2), ("Invert", 0.0, 3)),
        (("Equalize", 0.2, 0), ("AutoContrast", 0.6, 0)),
        (("Equalize", 0.2, 8), ("Equalize", 0.6, 4)),
        (("Color", 0.9, 9), ("Equalize", 0.6, 6)),
        (("AutoContrast", 0.8, 4), ("Solarize", 0.2, 8)),
        (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
        (("Solarize", 0.4, 5), ("AutoContrast", 0.9, 3)),
        (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
        (("AutoContrast", 0.9, 2), ("Solarize", 0.8, 3)),
        (("Equalize", 0.8, 8), ("Invert", 0.1, 3)),
        (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, 1)),
    ],
    "svhn": [
        (("ShearX", 0.9, 4), ("Invert", 0.2, 3)),
        (("ShearY", 0.9, 8), ("Invert", 0.7, 5)),
        (("Equalize", 0.6, 5), ("Solarize", 0.6, 6)),
        (("Invert", 0.9, 3), ("Equalize", 0.6, 3)),
        (("Equalize", 0.6, 1), ("Rotate", 0.9, 3)),
        (("ShearX", 0.9, 4), ("AutoContrast", 0.8, 3)),
        (("ShearY", 0.9, 8), ("Invert", 0.4, 5)),
        (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
        (("Invert", 0.9, 6), ("AutoContrast", 0.8, 1)),
        (("Equalize", 0.6, 3), ("Rotate", 0.9, 3)),
        (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
        (("ShearY", 0.8, 8), ("Invert", 0.7, 4)),
        (("Equalize", 0.9, 5), ("TranslateY", 0.6, 6)),
        (("Invert", 0.9, 4), ("Equalize", 0.6, 7)),
        (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
        (("Invert", 0.8, 5), ("TranslateY", 0.0, 2)),
        (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
        (("Invert", 0.6, 4), ("Rotate", 0.8, 4)),
        (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
        (("ShearX", 0.1, 6), ("Invert", 0.6, 5)),
        (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
        (("ShearY", 0.8, 4), ("Invert", 0.8, 8)),
        (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
        (("ShearY", 0.8, 5), ("AutoContrast", 0.7, 3)),
        (("ShearX", 0.7, 2), ("Invert", 0.1, 5)),
    ],
    "policy_v0": [
        (("Equalize", 0.8, 1), ("ShearY", 0.8, 4)),
        (("Color", 0.4, 9), ("Equalize", 0.6, 3)),
        (("Color", 0.4, 1), ("Rotate", 0.6, 8)),
        (("Solarize", 0.8, 3), ("Equalize", 0.4, 7)),
        (("Solarize", 0.4, 2), ("Solarize", 0.6, 2)),
        (("Color", 0.2, 0), ("Equalize", 0.8, 8)),
        (("Equalize", 0.4, 8), ("SolarizeAdd", 0.8, 3)),
        (("ShearX", 0.2, 9), ("Rotate", 0.6, 8)),
        (("Color", 0.6, 1), ("Equalize", 1.0, 2)),
        (("Invert", 0.4, 9), ("Rotate", 0.6, 0)),
        (("Equalize", 1.0, 9), ("ShearY", 0.6, 3)),
        (("Color", 0.4, 7), ("Equalize", 0.6, 0)),
        (("Posterize", 0.4, 6), ("AutoContrast", 0.4, 7)),
        (("Solarize", 0.6, 8), ("Color", 0.6, 9)),
        (("Solarize", 0.2, 4), ("Rotate", 0.8, 9)),
        (("Rotate", 1.0, 7), ("TranslateY", 0.8, 9)),
        (("ShearX", 0.0, 0), ("Solarize", 0.8, 4)),
        (("ShearY", 0.8, 0), ("Color", 0.6, 4)),
        (("Color", 1.0, 0), ("Rotate", 0.6, 2)),
        (("Equalize", 0.8, 4), ("Equalize", 0.0, 8)),
        (("Equalize", 1.0, 4), ("AutoContrast", 0.6, 2)),
        (("ShearY", 0.4, 7), ("SolarizeAdd", 0.6, 7)),
        (("Posterize", 0.8, 2), ("Solarize", 0.6, 10)),
        (("Solarize", 0.6, 8), ("Equalize", 0.6, 1)),
        (("Color", 0.8, 6), ("Rotate", 0.4, 5)),
    ],
}


class AutoAugment(PreprocessingLayer):
    """Implements AutoAugment to augment image data."""

    def __init__(self, policy: str, tf_function: bool = False, seed: int = 42):
        super().__init__()

        self.name_to_func = {
            "AutoContrast": AutoContrast(),
            "Brightness": Brightness(),
            "Color": Color(),
            "Contrast": Contrast(),
            "Cutout": CutOut(cutout_const=100),
            "Equalize": Equalize(),
            "Invert": Invert(),
            "Posterize": Posterize(),
            "Rotate": Rotate(),
            "Sharpness": Sharpness(),
            "ShearX": ShearX(),
            "ShearY": ShearY(),
            "Solarize": Solarize(),
            "SolarizeAdd": SolarizeAdd(),
            "TranslateX": TranslateX(translate_const=250),
            "TranslateY": TranslateY(translate_const=250),
        }
        self.rng = tf.random.Generator.from_seed(seed=seed)

        self.transforms = POLICIES[policy]
        self.num_transforms = len(self.transforms)

        if tf_function:
            self.call = tf.function(self.call)

    def _get_random_transform_idx(self) -> tf.Tensor:
        """Return random index for transforms pair in a given policy."""
        return self.rng.uniform(
            shape=(), minval=0, maxval=self.num_transforms, dtype=tf.int32
        )

    def _should_apply_transform(self, prob: tf.Tensor) -> bool:
        """Return true if policy should be applied.

        This is an equivalent of `bool(random > 1 - prob)`.
        """
        return tf.cast(
            tf.floor(self.rng.uniform((), minval=0, maxval=1, dtype=tf.float32) + prob),
            tf.bool,
        )

    def call(self, inputs, training=True, **kwargs):
        """Randomly apply transformations present in chosen policy."""
        if not training:
            return inputs

        transform_idx = self._get_random_transform_idx()

        for i in range(self.num_transforms):
            if i == transform_idx:  # Use int instead of Tensor.
                inputs = self._fetch_and_apply_policy(inputs, idx=i)

        return inputs

    def _fetch_and_apply_policy(self, inputs: tf.Tensor, idx: int) -> tf.Tensor:
        for name, prob, level in self.transforms[idx]:
            if self._should_apply_transform(tf.constant(prob)):
                func = self.name_to_func[name]
                inputs = func(inputs, tf.constant(level))
        return inputs
