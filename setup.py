"""ViLiS package setup."""

from setuptools import find_packages, setup

setup(
    name="vilis",
    version="0.1.0",
    description="ViLiS – Visual Linguistic Sync: Hindi lip-reading system",
    author="ViLiS Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "openai-whisper>=20231117",
        "sentencepiece>=0.1.99",
        "pyctcdecode>=0.5.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.16.0",
        "numpy",
        "tqdm",
        "av",
        "editdistance",
    ],
)
