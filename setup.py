from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup
from setuptools import find_packages

exclude_dirs = (
    "configs",
    "tests",
    "scripts",
    "data",
    "outputs",
)

setup(
    name="perec",
    version="0.1.0",
    author="Yaokun Xu",
    url="https://github.com/XU-YaoKun/perec",
    description="personalized recommendation in pytorch",
    packages=find_packages(exclude=exclude_dirs),
    install_requires=["torch", "numpy", "yacs", "tqdm"],
)
