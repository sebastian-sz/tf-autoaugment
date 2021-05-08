"""Benchmark Transforms on a batch of inputs."""
from typing import Dict

from absl import app, flags

from tf_autoaugment.benchmark.benchmarker import TransformBenchmarker
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

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", default=5, help="Batch size to use.", short_name="b")

ALL_TRANSFORMS = {
    "AutoContrast": AutoContrast,
    "Brightness": Brightness,
    "Color": Color,
    "Contrast": Contrast,
    "Cutout": CutOut,
    "Equalize": Equalize,
    "Invert": Invert,
    "Posterize": Posterize,
    "Rotate": Rotate,
    "Sharpness": Sharpness,
    "ShearX": ShearX,
    "ShearY": ShearY,
    "Solarize": Solarize,
    "SolarizeAdd": SolarizeAdd,
    "TranslateX": TranslateX,
    "TranslateY": TranslateY,
}


def main(argv):
    """Benchmark eager vs graph performance of transforms."""
    benchmarker = TransformBenchmarker(batch_size=FLAGS.batch_size)

    print("Running Eager benchmark:")
    summary: Dict[str, Dict[str, float]] = {}
    for transform_name in ALL_TRANSFORMS:
        print(f"Benchmarking {transform_name}")
        transform = ALL_TRANSFORMS[transform_name]()
        benchmarker.warmup(transform)
        time = benchmarker.run(transform)
        transform_summary = summary.get(transform_name, {})
        transform_summary.update({"eager_time": time})
        summary.update({transform_name: transform_summary})
        print(f"Took: {time}")

    print("Running tf-function benchmark:")
    for transform_name in ALL_TRANSFORMS:
        print(f"Benchmarking {transform_name}")
        transform = ALL_TRANSFORMS[transform_name](tf_function=True)
        benchmarker.warmup(transform)
        time = benchmarker.run(transform)
        transform_summary = summary.get(transform_name, {})
        transform_summary.update({"graph_time": time})
        summary.update({transform_name: transform_summary})
        print(f"Took: {time}")

    _print_summary(summary)


def _print_summary(summary: Dict[str, Dict[str, float]]):
    print("---------Summary info-------------")
    print("[Name] | [Eager Time] | [Graph Time]")
    for key, value in summary.items():
        print(f"{key}: {value['eager_time']:.5f} | {value['graph_time']:.5f}")
    print("----------------------------------")


if __name__ == "__main__":
    app.run(main)
