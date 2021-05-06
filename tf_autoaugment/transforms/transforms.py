import tensorflow as tf
import tensorflow_addons as tfa

from tf_autoaugment.image_ops import (
    autocontrast,
    brightness,
    color,
    contrast,
    cutout,
    equalize,
    invert,
    posterize,
    sharpness,
    shear_x,
    shear_y,
    solarize,
    solarize_add,
    translate_x,
    translate_y,
)
from tf_autoaugment.transforms.base_transform import BaseTransform

_MAX_LEVEL = 10.0


# Todo: add sources.
# Todo: add tf.function compatibility.
#   - tf.function can be enabled/disabled while creating autoaugment
#   - __call__ always accepts tf.Tensor elements. Or converts them apropriately.
#   - add tests assuring the function is not being retraced.


class Invert(BaseTransform):
    """Transform for invert function."""

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Invert input images."""
        return invert(images=images)

    def _parse_level(self, level: int) -> None:
        """Invert does not accept any additional arguments."""
        pass


class Solarize(BaseTransform):
    """Transform for Solarize op."""

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Call solarize on input images."""
        threshold = self._parse_level(level)
        return solarize(images=images, threshold=threshold)

    def _parse_level(self, level: int) -> int:
        return int((level / _MAX_LEVEL) * 256)


class Equalize(BaseTransform):
    """Transform for Equalize op."""

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        return equalize(images=images)

    def _parse_level(self, level: int) -> None:
        pass


class Posterize(BaseTransform):
    """Transform for Posterize op."""

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        original_dtype = images.dtype
        bits = self._parse_level(level)
        inputs = tf.cast(images, tf.uint8)
        outputs = posterize(images=inputs, bits=bits)
        return tf.cast(outputs, original_dtype)

    def _parse_level(self, level: int) -> int:
        return int((level / _MAX_LEVEL) * 4)


class Rotate(BaseTransform):
    """Transform for Rotate op."""

    def __init__(self, fill_value=128):
        super().__init__()
        self.fill_value = fill_value

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        angles = self._parse_level(level)
        return tfa.image.rotate(
            images=images, angles=angles, fill_value=self.fill_value
        )

    def _parse_level(self, level: int) -> tf.Tensor:
        level = (level / _MAX_LEVEL) * 30.0
        level = self._randomly_negate_tensor(tf.convert_to_tensor(level))
        return level

    def _debug_apply(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        angles = (level / _MAX_LEVEL) * 30.0
        return tfa.image.rotate(
            images=images, angles=angles, fill_value=self.fill_value
        )


class AutoContrast(BaseTransform):
    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        return autocontrast(images)  # Todo: test if non batched same

    def _parse_level(self, level: int) -> None:
        pass


class Brightness(BaseTransform):
    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        factor = self._parse_level(level)
        return brightness(images=images, factor=factor)

    def _parse_level(self, level: int) -> float:
        return (level / _MAX_LEVEL) * 1.8 + 0.1


class Color(BaseTransform):
    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        factor = self._parse_level(level)
        return color(images=images, factor=factor)

    def _parse_level(self, level: int) -> float:
        return (level / _MAX_LEVEL) * 1.8 + 0.1


class Contrast(BaseTransform):
    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        factor = self._parse_level(level)
        return contrast(images=images, factor=factor)  # Todo: test if non batched same

    def _parse_level(self, level: int) -> float:
        return (level / _MAX_LEVEL) * 1.8 + 0.1


class Sharpness(BaseTransform):
    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        factor = self._parse_level(level)
        return sharpness(images=images, factor=factor)  # Todo: test if non batched same

    def _parse_level(self, level: int) -> float:
        # Todo: move this to base class as enhance level. Add source.
        return (level / _MAX_LEVEL) * 1.8 + 0.1


class ShearX(BaseTransform):
    def __init__(self, fill_value=128):
        super().__init__()

        if isinstance(fill_value, int):
            self.fill_value = tf.constant([fill_value] * 3, dtype=tf.int32)
        elif isinstance(fill_value, list) or isinstance(fill_value, tuple):
            assert len(fill_value) == 3, "fill_value must be int or 3 element sequence"
            self.fill_value = tf.constant(fill_value, dtype=tf.int32)

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        level = self._parse_level(level)
        return shear_x(images=images, level=level, replace=self.fill_value)

    def _parse_level(self, level: int) -> float:
        level = (level / _MAX_LEVEL) * 0.3
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return level

    def _debug_apply(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        level = (level / _MAX_LEVEL) * 0.3
        return shear_x(images=images, level=level, replace=self.fill_value)


class ShearY(BaseTransform):
    def __init__(self, fill_value=128):
        super().__init__()

        if isinstance(fill_value, int):
            self.fill_value = tf.constant([fill_value] * 3, dtype=tf.int32)
        elif isinstance(fill_value, list) or isinstance(fill_value, tuple):
            assert len(fill_value) == 3, "fill_value must be int or 3 element sequence"
            self.fill_value = tf.constant(fill_value, dtype=tf.int32)

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        level = self._parse_level(level)
        return shear_y(images=images, level=level, replace=self.fill_value)

    def _parse_level(self, level: int) -> float:
        level = (level / _MAX_LEVEL) * 0.3
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return level

    def _debug_apply(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        level = (level / _MAX_LEVEL) * 0.3
        return shear_y(images=images, level=level, replace=self.fill_value)


class TranslateX(BaseTransform):
    def __init__(self, fill_value=128, translate_const=250):
        super().__init__()
        self.translate_const = translate_const

        if isinstance(fill_value, int):
            self.fill_value = tf.constant([fill_value] * 3, dtype=tf.int32)
        elif isinstance(fill_value, list) or isinstance(fill_value, tuple):
            assert len(fill_value) == 3, "fill_value must be int or 3 element sequence"
            self.fill_value = tf.constant(fill_value, dtype=tf.int32)

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        level = self._parse_level(level)
        return translate_x(images=images, replace=self.fill_value, pixels=level)

    def _parse_level(self, level: int) -> float:
        level = (level / _MAX_LEVEL) * float(self.translate_const)
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return level

    def _debug_apply(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        level = (level / _MAX_LEVEL) * float(self.translate_const)
        return translate_x(images=images, replace=self.fill_value, pixels=level)


class TranslateY(BaseTransform):
    def __init__(self, fill_value=128, translate_const=250):
        super().__init__()
        self.translate_const = translate_const

        if isinstance(fill_value, int):
            self.fill_value = tf.constant([fill_value] * 3, dtype=tf.int32)
        elif isinstance(fill_value, list) or isinstance(fill_value, tuple):
            assert len(fill_value) == 3, "fill_value must be int or 3 element sequence"
            self.fill_value = tf.constant(fill_value, dtype=tf.int32)

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        level = self._parse_level(level)
        return translate_y(images=images, replace=self.fill_value, pixels=level)

    def _parse_level(self, level: int) -> float:
        level = (level / _MAX_LEVEL) * float(self.translate_const)
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return level

    def _debug_apply(self, images: tf.Tensor, level: int) -> tf.Tensor:
        """Only for testing. Call transform without self._randomly_negate_tensor."""
        level = (level / _MAX_LEVEL) * float(self.translate_const)
        return translate_y(images=images, replace=self.fill_value, pixels=level)


class SolarizeAdd(BaseTransform):
    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        addition = self._parse_level(level)
        return solarize_add(
            images=images, addition=addition, threshold=128
        )  # Todo: make sure this is ok.

    def _parse_level(self, level: int) -> int:
        return int((level / _MAX_LEVEL) * 110)


class CutOut(BaseTransform):
    def __init__(self, fill_value=128, cutout_const=100):
        super().__init__()
        self.cutout_const = cutout_const

        if isinstance(fill_value, int):
            self.fill_value = tf.constant([fill_value] * 3, dtype=tf.uint8)
        elif isinstance(fill_value, list) or isinstance(fill_value, tuple):
            assert len(fill_value) == 3, "fill_value must be int or 3 element sequence"
            self.fill_value = tf.constant(fill_value, dtype=tf.uint8)

    def __call__(self, images: tf.Tensor, level: int) -> tf.Tensor:
        pad_size = self._parse_level(level)
        return cutout(
            image=images, pad_size=pad_size, replace=self.fill_value
        )  # Todo: make sure this is ok.

    def _parse_level(self, level: int) -> int:
        return int((level / _MAX_LEVEL) * 110)

    def _debug_apply(self, images: tf.Tensor, level: int) -> tf.Tensor:
        pad_size = self._parse_level(level)
        return self._debug_cutout(images, pad_size, self.fill_value)

    @staticmethod
    def _debug_cutout(image, pad_size, replace=0):
        """Only for testing. Cutout always the same part of the image."""
        image_height = tf.shape(image)[1]
        image_width = tf.shape(image)[2]

        replace = tf.cast(replace, image.dtype)

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = 10
        cutout_center_width = 10

        lower_pad = tf.maximum(0, cutout_center_height - pad_size)
        upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
        left_pad = tf.maximum(0, cutout_center_width - pad_size)
        right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

        cutout_shape = [
            image_height - (lower_pad + upper_pad),
            image_width - (left_pad + right_pad),
        ]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = tf.pad(
            tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
        )
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 3])
        image = tf.where(
            tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image
        )
        return image


# Todo: maybe integrate image ops directly into transform?
NAME_TO_TRANSFORM = {
    "Invert": Invert(),
    "Solarize": Solarize(),
    "Equalize": Equalize(),
    "Posterize": Posterize(),
    "Rotate": Rotate(),
    "Autocontrast": AutoContrast(),
    "Brightness": Brightness(),
    "Color": Color(),
    "Contrast": Contrast(),
    "Sharpness": Sharpness(),
    "ShearX": ShearX(),
    "ShearY": ShearY(),
    "TranslateX": TranslateX(),
    "TranslateY": TranslateY(),
    "SolarizeAdd": SolarizeAdd(),
    "Cutout": CutOut(),
}
