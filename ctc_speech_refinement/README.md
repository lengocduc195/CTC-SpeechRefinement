# CTC Speech Refinement

A speech recognition system using CTC decoding with speculative decoding and consistency regularization.

## Project Structure

```
ctc_speech_refinement/
├── core/                   # Core functionality
│   ├── preprocessing/      # Audio preprocessing
│   ├── features/           # Feature extraction
│   ├── models/             # Acoustic models
│   ├── decoder/            # CTC decoder implementations
│   ├── utils/              # Utility functions
│   └── eda/                # Exploratory data analysis
├── apps/                   # Applications
│   ├── transcription/      # Basic transcription
│   ├── speculative_decoding/ # Speculative decoding
│   ├── audio_eda/          # Audio EDA tools
│   └── ui/                 # User interfaces
├── config/                 # Configuration
├── tests/                  # Unit tests
└── setup.py                # Package setup
```

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CTC-SpeechRefinement.git
cd CTC-SpeechRefinement
```

2. Install the package in development mode:
```bash
pip install -e ctc_speech_refinement
```

## Usage

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
from ctc_speech_refinement.apps.speculative_decoding.decoder import SpeculativeDecoder

# Initialize speculative decoder
decoder = SpeculativeDecoder()

# Transcribe audio
result = decoder.transcribe("path/to/audio.wav")
print(f"Transcription: {result['transcription']}")
print(f"Processing time: {result['total_time_ms']:.2f} ms")
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

### Audio EDA

```python
from ctc_speech_refinement.apps.audio_eda.analyzer import analyze_audio_file

# Analyze audio file
results = analyze_audio_file(
    "path/to/audio.wav",
    output_dir="results/eda",
    normalize=True,
    remove_silence=True
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

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [Audio EDA Guide](docs/AUDIO_EDA_GUIDE.md)
- [Speculative Decoding Guide](docs/SPECULATIVE_DECODING.md)
- [Integration Guide](docs/INTEGRATION_GUIDE.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
