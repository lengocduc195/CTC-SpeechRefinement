"""
Configuration settings for the CTC Speech Transcription project.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data directories
DATA_DIR = ROOT_DIR / "data"
TEST1_DIR = DATA_DIR / "test1"
TEST2_DIR = DATA_DIR / "test2"

# Output directories
RESULTS_DIR = ROOT_DIR / "results"
TRANSCRIPTS_DIR = ROOT_DIR / "transcripts"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for directory in [RESULTS_DIR, TRANSCRIPTS_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# Audio preprocessing settings
SAMPLE_RATE = 16000  # Target sample rate in Hz
AUDIO_DURATION = None  # Set to None to use full audio or a value in seconds to trim

# Feature extraction settings
FEATURE_TYPE = "mfcc"  # Options: "mfcc", "mel_spectrogram", "spectrogram"
N_MFCC = 40  # Number of MFCC features
N_MELS = 128  # Number of Mel bands
N_FFT = 512  # FFT window size
HOP_LENGTH = 160  # Hop length for STFT (10ms at 16kHz)
WIN_LENGTH = 400  # Window length for STFT (25ms at 16kHz)

# Model settings
MODEL_TYPE = "pretrained"  # Options: "pretrained", "custom"
PRETRAINED_MODEL_NAME = "facebook/wav2vec2-base-960h"  # Pretrained model from Hugging Face

# CTC Decoder settings
DECODER_TYPE = "greedy"  # Options: "greedy", "beam_search"
BEAM_WIDTH = 100  # Beam width for beam search decoding
ALPHA = 0.5  # Language model weight
BETA = 1.0  # Word insertion bonus

# Language model settings
USE_LM = False  # Whether to use a language model
LM_PATH = None  # Path to language model file

# Evaluation settings
COMPUTE_WER = True  # Whether to compute Word Error Rate
COMPUTE_CER = True  # Whether to compute Character Error Rate
