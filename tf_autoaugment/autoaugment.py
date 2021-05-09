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

POLICIES = {"_debug": [[("TranslateX", 1.0, 4), ("Equalize", 1.0, 10)]]}


class AutoAugment(PreprocessingLayer):
    """Implements AutoAugment to augment image data."""

    def __init__(self, policy: str, tf_function: bool = False, seed: int = 42):
        super().__init__()

        self.name_to_func = {
            "AutoContrast": AutoContrast(),
            "Brightness": Brightness(),
            "Color": Color(),
            "Contrast": Contrast(tf_function),
            "Cutout": CutOut(),  # Todo: unused.
            "Equalize": Equalize(),
            "Invert": Invert(),
            "Posterize": Posterize(),
            "Rotate": Rotate(),
            "Sharpness": Sharpness(),
            "ShearX": ShearX(),
            "ShearY": ShearY(),
            "Solarize": Solarize(),
            "SolarizeAdd": SolarizeAdd(),  # Todo: unused.
            "TranslateX": TranslateX(),
            "TranslateY": TranslateY(),
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
