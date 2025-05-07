"""
Setup script for the CTC Speech Refinement package.
"""

from setuptools import setup, find_packages

setup(
    name="ctc_speech_refinement",
    version="0.1.0",
    description="Speech recognition system using CTC decoding with speculative decoding and consistency regularization",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "transformers>=4.18.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "soundfile>=0.10.3",
        "pyctcdecode>=0.3.0",
        "jiwer>=2.3.0",
        "noisereduce>=2.0.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
