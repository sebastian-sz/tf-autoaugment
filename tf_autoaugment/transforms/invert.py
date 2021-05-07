"""Code for Invert Transform."""
import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform


class Invert(BaseTransform):
    """Invert Transform using TF Ops."""

    def __call__(self, images: tf.Tensor, level: tf.Tensor) -> tf.Tensor:
        """Run invert function on image batch."""
        return self.invert(images)

    @staticmethod
    def invert(images: tf.Tensor) -> tf.Tensor:
        """Invert the pixels of the image."""
        return 255 - images

    def _parse_level(self, level: tf.Tensor) -> tf.Tensor:
        """Level is unused by this op."""
        pass
