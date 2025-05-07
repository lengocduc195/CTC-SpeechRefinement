"""
Default configuration settings for the CTC Speech Refinement project.
"""

import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = Path(os.path.dirname(ROOT_DIR))

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TEST1_DIR = DATA_DIR / "test1"
TEST2_DIR = DATA_DIR / "test2"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
MODELS_DIR = PROJECT_ROOT / "models"

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

# Speculative decoding settings
DRAFTER_MODEL_NAME = "facebook/wav2vec2-base-960h"
VERIFIER_MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
MAX_DRAFT_LENGTH = 100
DRAFT_TIMEOUT_MS = 500
ACCEPTANCE_THRESHOLD = 0.5

# CR-CTC settings
USE_CR_CTC = True
NUM_PERTURBATIONS = 3
FALLBACK_TO_STANDARD = True

# Audio preprocessing options
NORMALIZE_AUDIO = True
REMOVE_SILENCE = True
APPLY_VAD = False
VAD_METHOD = "energy"
REDUCE_NOISE = False
NOISE_REDUCTION_METHOD = "spectral_subtraction"
NORMALIZE_FREQUENCY = False
FREQUENCY_NORMALIZATION_METHOD = "bandpass"
