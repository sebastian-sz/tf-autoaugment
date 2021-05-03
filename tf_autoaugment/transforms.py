"""Contains Code for all transforms present in AutoAugment operation."""
import math
from abc import ABC, abstractmethod
from typing import Union

import tensorflow as tf
import tensorflow_addons as tfa

from tf_autoaugment.raw_img_ops import (
    batched_autocontrast,
    batched_equalize,
    batched_shear_x,
    batched_shear_y,
    invert,
    posterize,
    solarize,
    translate_x,
    translate_y,
)


class BaseTransform(ABC):
    """Base class for AutoAugment transforms."""

    @abstractmethod
    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Run transform on provided images batch.

        :param images: images tensor of shape (batch, height, width, channels)
        :param magnitude: float tensor.
        :return: transformed images tensor of the same shape as input.
        """
        raise NotImplementedError


class Rotate(BaseTransform):
    """Implements Autoaugment's Rotate transform."""

    def __init__(self, fill_value=128):
        super().__init__()
        self.fill_value = fill_value

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Randomly rotates image batch by angle specified via magnitude."""
        return tfa.image.rotate(images, angles=magnitude, fill_value=self.fill_value)


class Color(BaseTransform):
    """Implements Autoaugment's Color transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Change images saturation based on provided magnitude."""
        saturation_factor = 1.0 + magnitude
        return tf.image.adjust_saturation(images, saturation_factor=saturation_factor)


class Brightness(BaseTransform):
    """Implements Autoaugment's Brightness transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Change images brightness based on provided magnitude."""
        delta = 1.0 + magnitude
        return tf.image.adjust_brightness(images, delta=delta)


class Posterize(BaseTransform):
    """Implements Autoaugment's Posterize transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Shift images values by the amount of bits based on magnitude."""
        bits = int(magnitude)
        return posterize(images, bits=bits)


class Contrast(BaseTransform):
    """Implements Autoaugment's Contrast transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Adjust images contrast based on provided magnitude."""
        contrast_factor = 1.0 + magnitude
        return tf.image.adjust_contrast(images, contrast_factor=contrast_factor)


class Sharpness(BaseTransform):
    """Implements Autoaugment's Sharpness transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Change sharpness of images based on provided magnitude."""
        factor = 1.0 + magnitude
        return tfa.image.sharpness(images, factor=factor)


class ShearX(BaseTransform):
    """Implements Autoaugment's ShearX transform."""

    def __init__(self, fill_value=128):
        super().__init__()
        self.fill_value = fill_value

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Apply shear_x transform with a level based on provided magnitude."""
        level = math.degrees(magnitude)
        return batched_shear_x(images, level=level, replace=self.fill_value)


class ShearY(BaseTransform):
    """Implements Autoaugment's ShearY transform."""

    def __init__(self, fill_value=128):
        super().__init__()
        self.fill_value = fill_value

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Apply shear_y transform with a level based on provided magnitude."""
        level = math.degrees(magnitude)
        return batched_shear_y(images, level=level, replace=self.fill_value)


class TranslateX(BaseTransform):
    """Implements Autoaugment's TranslateX transform."""

    def __init__(self, fill_value=128):
        super().__init__()
        self.fill_value = fill_value

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Run translate_x operation on a batch of images."""
        pixels = images.shape[1]
        return translate_x(images, pixels=pixels, replace=self.fill_value)


class TranslateY(BaseTransform):
    """Implements Autoaugment's TranslateY transform."""

    def __init__(self, fill_value=128):
        super().__init__()
        self.fill_value = fill_value

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Run translate_y operation on a batch of images."""
        pixels = images.shape[2]
        return translate_y(images, pixels=pixels, replace=self.fill_value)


class Invert(BaseTransform):
    """Implements Autoaugment's Invert transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Invert images values."""
        return invert(images)


class Solarize(BaseTransform):
    """Implements Autoaugment's Solarize transformation."""

    def __init__(self, threshold=128):
        super().__init__()
        self.threshold = threshold

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Run solarize transform on provided images."""
        return solarize(images, threshold=self.threshold)


class AutoContrast(BaseTransform):
    """Implements Autoaugment's AutoContrast transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Run autocontrast on a batch of images."""
        return batched_autocontrast(images)


class Equalize(BaseTransform):
    """Implements Autoaugment's Equalize transform."""

    def __call__(
        self, images: tf.Tensor, magnitude: Union[tf.Tensor, float]
    ) -> tf.Tensor:
        """Run equalize on a batch of images."""
        return batched_equalize(images)


NAME_TO_TRANSFORM = {
    "Rotate": Rotate(),
    "Solarize": Solarize(),
    "Color": Color(),
    "Invert": Invert(),
    "Contrast": Contrast(),
    "Brightness": Brightness(),
    "Posterize": Posterize(),
    "AutoContrast": AutoContrast(),
    "Equalize": Equalize(),
    "Sharpness": Sharpness(),
    "ShearX": ShearX(),
    "ShearY": ShearY(),
    "TranslateX": TranslateX(),
    "TranslateY": TranslateY(),
}
