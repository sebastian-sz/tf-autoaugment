"""Code for Transform Benchmarker."""
import timeit

import tensorflow as tf

from tf_autoaugment.transforms.base_transform import BaseTransform


class TransformBenchmarker:
    """Utility class for benchmarking transforms execution time."""

    RNG = tf.random.Generator.from_non_deterministic_state()

    def __init__(self, batch_size: int = 5):
        self.inputs = self.RNG.uniform(
            shape=(batch_size, 224, 224, 3), minval=0, maxval=255, dtype=tf.int32
        )
        self.level = tf.constant(3)

    def run(self, transform: BaseTransform):
        """Benchmark transform execution time."""
        return timeit.timeit(lambda: transform(self.inputs, self.level), number=100)

    def warmup(self, transform: BaseTransform):
        """Run transform once to warmup."""
        transform(self.inputs, self.level)
