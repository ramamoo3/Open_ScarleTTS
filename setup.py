from setuptools import setup, find_packages

setup(
    name="open_scarletts",
    version="0.1.0",
    description="Lightweight Kokoro-ONNX wrapper with runtime emotion styling",
    author="ramamoo3",
    packages=find_packages(),
    install_requires=[
        "kokoro-onnx>=0.3.0",
        "sounddevice>=0.4.6",
        "numpy>=1.22.0",
        "soundfile>=0.12.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
