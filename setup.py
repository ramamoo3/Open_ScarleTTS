import re
from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).parent / "README.md"
INIT = Path(__file__).parent / "open_scarletts" / "__init__.py"

version = re.search(r'^__version__\s*=\s*"([^"]+)"', INIT.read_text(), re.M).group(1)

setup(
    name="open_scarletts",
    version=version,
    description="Lightweight Kokoro-ONNX wrapper with runtime emotion styling",
    long_description=README.read_text(),
    long_description_content_type="text/markdown",
    author="ramamoo3",
    url="https://github.com/ramamoo3/Open_ScarleTTS",
    license="MIT",
    keywords="tts text-to-speech kokoro onnx emotions edge raspberry-pi speech-synthesis",
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "kokoro-onnx>=0.3.0",
        "sounddevice>=0.4.6",
        "numpy>=1.22.0",
        "soundfile>=0.12.1",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "scarletts=open_scarletts.cli:main",
            "scarletts-bench=open_scarletts.bench:main",
            "scarletts-setup=open_scarletts.assets:cli_setup",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
