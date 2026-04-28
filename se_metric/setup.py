"""Setup script for the se-metric package."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="se-metric",
    version="0.1.0",
    description="Structure Error (SE) metric for symbolic music evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anon/se-metric",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "miditoolkit>=0.1.16",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
    ],
)
