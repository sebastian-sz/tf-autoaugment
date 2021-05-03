"""Package installation file."""

from setuptools import find_packages, setup

setup(
    name="tf-autoaugment",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.6",
    description="Tensorflow ops for AutoAugment.",
)
