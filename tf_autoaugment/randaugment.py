# type: ignore
"""Code for RandAugment operation."""
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


class RandAugment(PreprocessingLayer):
    """Implements RandAugment to augment image data."""

    def __init__(
        self,
        num_layers: int = 2,
        magnitude: int = 15,
        tf_function: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.magnitude = tf.constant(magnitude)

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
        self.available_ops = list(self.name_to_func.keys())

        self.num_transforms = len(self.name_to_func)
        self.rng = tf.random.Generator.from_seed(seed=seed)

        if tf_function:
            self.call = tf.function(self.call)

    def call(self, inputs, training=True, **kwargs):
        """Randomly choose and apply a transform, present in available ops."""
        for layer_num in range(self.num_layers):
            op_to_select = self._random_int()
            for i in range(self.num_transforms):

                if i == op_to_select:
                    tf.print(i)

                    name = self.available_ops[i]
                    tf.print(name)
                    func = self.name_to_func[name]
                    inputs = func(inputs, self.magnitude)

        return inputs

    def _random_int(self):
        return self.rng.uniform(
            shape=(), minval=0, maxval=self.num_transforms, dtype=tf.int32
        )
