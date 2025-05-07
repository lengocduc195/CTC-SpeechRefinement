# Quick Start Guide

This guide provides a quick introduction to the CTC Speech Refinement system.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CTC-SpeechRefinement.git
cd CTC-SpeechRefinement
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e ctc_speech_refinement
```

## Project Structure

The project is organized as follows:

```
CTC-SpeechRefinement/
├── ctc_speech_refinement/     # Main package
│   ├── core/                  # Core functionality
│   ├── apps/                  # Applications
│   ├── config/                # Configuration
│   └── tests/                 # Unit tests
├── data/                      # Audio data
├── docs/                      # Documentation
├── models/                    # Saved models
├── results/                   # Evaluation results
├── transcripts/               # Generated transcriptions
└── [Various scripts]          # Scripts to run the applications
```

## Basic Usage

### Transcribing Speech

To transcribe speech using the basic transcription system:

```bash
python run_transcription.py --input_dir data/test1 --output_dir transcripts
```

### Speculative Decoding

To use speculative decoding for faster and more accurate transcription:

```bash
python run_speculative_decoding.py --input_dir data/test1 --output_dir transcripts --results_dir results
```

### Audio EDA

To analyze audio data:

```bash
python run_audio_eda_new.py --input data/test1 --output_dir results/eda
```

### Audio Preprocessing UI

To use the audio preprocessing UI:

```bash
python run_preprocessing_ui_new.py --language vi
```

## Python API Examples

### Basic Transcription

```python
from ctc_speech_refinement.core.preprocessing.audio import preprocess_audio
from ctc_speech_refinement.core.models.acoustic_model import AcousticModel
from ctc_speech_refinement.core.decoder.ctc_decoder import CTCDecoder

# Load and preprocess audio
audio_data, sample_rate = preprocess_audio("path/to/audio.wav")

# Initialize model and decoder
model = AcousticModel()
decoder = CTCDecoder(model.processor)

# Get logits from model
logits = model.get_logits(audio_data, sample_rate)

# Decode to get transcription
transcription = decoder.decode(logits)
print(f"Transcription: {transcription}")
```

### Speculative Decoding

```python
from ctc_speech_refinement.core.preprocessing.audio import preprocess_audio
from ctc_speech_refinement.apps.speculative_decoding.decoder import SpeculativeDecoder

# Load and preprocess audio
audio_data, sample_rate = preprocess_audio("path/to/audio.wav")

# Initialize speculative decoder
decoder = SpeculativeDecoder()

# Perform speculative decoding
results = decoder.decode(audio_data, sample_rate)
print(f"Transcription: {results['transcription']}")
print(f"Processing time: {results['total_time_ms']:.2f} ms")
```

### Audio Preprocessing

```python
from ctc_speech_refinement.core.preprocessing.audio import preprocess_audio

# Preprocess audio with various options
audio_data, sample_rate = preprocess_audio(
    "path/to/audio.wav",
    normalize=True,
    remove_silence_flag=True,
    apply_vad_flag=True,
    vad_method="energy",
    reduce_noise_flag=True,
    noise_reduction_method="spectral_subtraction",
    normalize_frequency_flag=True,
    frequency_normalization_method="bandpass"
)
```

## Configuration

The package uses a configuration system that can be customized:

```python
from ctc_speech_refinement.config.config_loader import load_config

# Load default configuration
config = load_config()

# Load custom configuration
config = load_config("path/to/config.json")

# Access configuration values
sample_rate = config["SAMPLE_RATE"]
```

## Documentation

For more detailed documentation, see the following guides:

- [Running the Code](RUNNING_THE_CODE.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Audio EDA Guide](AUDIO_EDA_GUIDE.md)
- [Speculative Decoding Guide](SPECULATIVE_DECODING.md)
- [Integration Guide](INTEGRATION_GUIDE.md)
- [Audio Preprocessing Guide](AUDIO_PREPROCESSING_GUIDE.md)
- [Audio Preprocessing Guide (Vietnamese)](AUDIO_PREPROCESSING_GUIDE_VI.md)

## Getting Help

If you encounter any issues, please check the logs for error messages and refer to the API documentation for more information.
