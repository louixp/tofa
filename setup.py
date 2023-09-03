import os

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="tofa",
    description="Torch Functional (Linear) Algebra",
    author="Yilong Qin",
    author_email="yilongq@cs.cmu.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/louixp/tofa",
    packages=find_packages(),
    include_package_data=True,
    version="0.0.1",
    install_requires=["numpy", "torch"],
    extras_require={
        "develop": ["pytest-cov", "black", "isort", "pytype"],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
    ],
)
