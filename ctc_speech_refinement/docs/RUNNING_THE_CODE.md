# Running the Code

This guide provides detailed instructions for running the various components of the CTC Speech Refinement system.

## Prerequisites

Before running the code, make sure you have:

1. Installed the package in development mode:
   ```bash
   pip install -e ctc_speech_refinement
   ```

2. Downloaded the necessary data and models (if applicable).

## Basic Transcription

The basic transcription system uses a single acoustic model with CTC decoding to transcribe speech.

### Command-Line Interface

```bash
python run_transcription.py --input_dir data/test1 --output_dir transcripts
```

#### Options

- `--input_dir`: Directory containing audio files to transcribe
- `--output_dir`: Directory to save transcriptions
- `--results_dir`: Directory to save results
- `--model_name`: Pretrained model name or path (default: "facebook/wav2vec2-base-960h")
- `--decoder_type`: Type of CTC decoder to use (choices: "greedy", "beam_search")
- `--beam_width`: Beam width for beam search decoding
- `--normalize_audio`: Normalize audio data
- `--remove_silence`: Remove silent regions from audio
- `--reference_dir`: Directory containing reference transcriptions for evaluation

### Python API

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

## Speculative Decoding

Speculative decoding uses a smaller, faster model to generate draft transcriptions that are then verified by a larger, more accurate model.

### Command-Line Interface

```bash
python run_speculative_decoding.py --input_dir data/test1 --output_dir transcripts --results_dir results
```

#### Options

See the [Speculative Decoding Guide](SPECULATIVE_DECODING.md#command-line-interface) for a complete list of options.

### Python API

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

## Audio EDA (Exploratory Data Analysis)

The Audio EDA system provides tools for analyzing audio data.

### Command-Line Interface

```bash
python run_audio_eda_new.py --input data/test1 --output_dir results/eda
```

#### Options

- `--input`: Path to audio file or directory
- `--output_dir`: Directory to save results
- `--normalize`: Normalize audio data
- `--remove_silence`: Remove silent regions from audio
- `--plot_format`: Format for saving plots (choices: "png", "pdf", "svg")
- `--dpi`: DPI for saving plots
- `--save_audio`: Save preprocessed audio files

### Python API

```python
from ctc_speech_refinement.apps.audio_eda.analyzer import analyze_audio_file, analyze_directory

# Analyze a single file
results = analyze_audio_file(
    "path/to/audio.wav",
    output_dir="results/eda",
    normalize=True,
    remove_silence=True
)

# Analyze all files in a directory
results = analyze_directory(
    "path/to/directory",
    output_dir="results/eda",
    normalize=True,
    remove_silence=True
)
```

## Audio Preprocessing UI

The Audio Preprocessing UI provides a graphical interface for configuring and applying preprocessing options to audio files.

### Command-Line Interface

```bash
python run_preprocessing_ui_new.py --language vi
```

#### Options

- `--language`: UI language (choices: "en", "vi", default: "vi")

### Python API

```python
from ctc_speech_refinement.apps.ui.preprocessing_ui import PreprocessingUI
import tkinter as tk

# Create Tkinter root window
root = tk.Tk()

# Initialize preprocessing UI
app = PreprocessingUI(root, language="vi")

# Run the UI
root.mainloop()
```

## Audio Preprocessing

The audio preprocessing module provides various techniques to improve the quality of audio data.

### Python API

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

#### Voice Activity Detection (VAD)

```python
from ctc_speech_refinement.core.preprocessing.vad import apply_vad, energy_vad, zcr_vad
import librosa

# Load audio
audio_data, sample_rate = librosa.load("path/to/audio.wav", sr=16000)

# Apply VAD
speech_audio = apply_vad(audio_data, sample_rate, method="energy")

# Get speech regions
speech_regions = energy_vad(audio_data, sample_rate)
for start_time, end_time in speech_regions:
    print(f"Speech from {start_time:.2f}s to {end_time:.2f}s")
```

#### Noise Reduction

```python
from ctc_speech_refinement.core.preprocessing.noise_reduction import reduce_noise
import librosa

# Load audio
audio_data, sample_rate = librosa.load("path/to/audio.wav", sr=16000)

# Apply noise reduction
denoised_audio = reduce_noise(
    audio_data, 
    sample_rate, 
    method="spectral_subtraction"
)
```

#### Frequency Normalization

```python
from ctc_speech_refinement.core.preprocessing.frequency_normalization import normalize_frequency
import librosa

# Load audio
audio_data, sample_rate = librosa.load("path/to/audio.wav", sr=16000)

# Apply frequency normalization
normalized_audio = normalize_frequency(
    audio_data, 
    sample_rate, 
    method="bandpass"
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

### Creating a Custom Configuration

You can create a custom configuration file in JSON format:

```json
{
    "SAMPLE_RATE": 16000,
    "AUDIO_DURATION": null,
    "NORMALIZE_AUDIO": true,
    "REMOVE_SILENCE": true,
    "APPLY_VAD": true,
    "VAD_METHOD": "energy",
    "REDUCE_NOISE": true,
    "NOISE_REDUCTION_METHOD": "spectral_subtraction",
    "NORMALIZE_FREQUENCY": true,
    "FREQUENCY_NORMALIZATION_METHOD": "bandpass"
}
```

Or in Python format:

```python
"""
Custom configuration for the CTC Speech Refinement project.
"""

# Audio preprocessing settings
SAMPLE_RATE = 16000
AUDIO_DURATION = None
NORMALIZE_AUDIO = True
REMOVE_SILENCE = True
APPLY_VAD = True
VAD_METHOD = "energy"
REDUCE_NOISE = True
NOISE_REDUCTION_METHOD = "spectral_subtraction"
NORMALIZE_FREQUENCY = True
FREQUENCY_NORMALIZATION_METHOD = "bandpass"
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure you have installed the package in development mode:
   ```bash
   pip install -e ctc_speech_refinement
   ```

2. **ModuleNotFoundError**: Check that you're running the scripts from the project root directory.

3. **Missing Dependencies**: Make sure all dependencies are installed:
   ```bash
   pip install -r ctc_speech_refinement/requirements.txt
   ```

4. **GPU Issues**: If you're using GPU acceleration, make sure you have the correct CUDA version installed for your PyTorch version.

5. **Audio File Format Issues**: The system supports WAV, MP3, FLAC, and OGG formats. Make sure your audio files are in one of these formats.

### Getting Help

If you encounter any issues, please check the logs for error messages and refer to the API documentation for more information.
