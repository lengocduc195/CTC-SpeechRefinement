# Audio Exploratory Data Analysis (EDA) Guide

This guide explains how to use the audio EDA module to analyze audio files and extract insights from them.

## Overview

The audio EDA module provides comprehensive analysis of audio data, including:

1. **Descriptive Statistics Analysis**: Basic statistical measures of audio amplitude
2. **Time Domain Analysis**: Waveform, envelope, energy, and zero crossing analysis
3. **Frequency Domain Analysis**: FFT, spectrograms, and spectral features
4. **Pitch and Timbre Analysis**: Fundamental frequency and MFCC features
5. **Anomaly Detection**: Identification of unusual patterns in audio

## Usage

### Command-Line Interface

The easiest way to use the audio EDA module is through the command-line interface:

```bash
python run_audio_eda.py --input data/test1 --output_dir results/eda --normalize --remove_silence
```

#### Basic Options

- `--input`: Path to an audio file or directory containing audio files
- `--output_dir`: Directory to save analysis results and plots
- `--normalize`: Normalize audio data to have zero mean and unit variance
- `--remove_silence`: Remove silent regions from audio
- `--single_file`: Treat input as a single file even if it's a directory

#### Advanced Preprocessing Options

- `--trim_start`: Start time for trimming in seconds (default: 0.0)
- `--trim_end`: End time for trimming in seconds (default: None)
- `--fade_in`: Fade-in time in seconds (default: 0.0)
- `--fade_out`: Fade-out time in seconds (default: 0.0)
- `--filter_type`: Type of filter to apply (choices: "lowpass", "highpass", "bandpass", "bandstop")
- `--cutoff_freq`: Cutoff frequency for the filter in Hz (default: 1000.0)
- `--cutoff_low`: Lower cutoff frequency for bandpass/bandstop filter in Hz (default: 500.0)
- `--cutoff_high`: Upper cutoff frequency for bandpass/bandstop filter in Hz (default: 2000.0)
- `--add_noise`: Level of noise to add (standard deviation) (default: None)

### Python API

You can also use the audio EDA module directly in your Python code:

```python
from src.eda.audio_eda import analyze_audio_file, analyze_directory

# Analyze a single file
results = analyze_audio_file(
    "data/test1/test1_01.wav",
    output_dir="results/eda",
    normalize=True,
    remove_silence=True
)

# Analyze all files in a directory
results = analyze_directory(
    "data/test1",
    output_dir="results/eda",
    normalize=True,
    remove_silence=True
)
```

## Output

The audio EDA module generates the following outputs:

### Plots

1. **Descriptive Statistics**
   - Amplitude distribution
   - Statistics summary

2. **Time Domain Analysis**
   - Waveform
   - Envelope
   - Energy
   - Zero crossing rate
   - Silence detection

3. **Frequency Domain Analysis**
   - FFT magnitude spectrum
   - Spectrogram
   - Mel spectrogram
   - Chromagram
   - Spectral contrast
   - Spectral features (centroid, bandwidth, rolloff, flatness)

4. **Pitch and Timbre Analysis**
   - Pitch estimation
   - Pitch distribution
   - MFCC features
   - MFCC statistics

5. **Anomaly Detection**
   - Amplitude anomalies
   - Spectral anomalies

### Analysis Results

The analysis results are saved in JSON format for each file, containing:

- Basic information (file path, sample rate, duration)
- Descriptive statistics (mean, std, min, max, etc.)
- Time domain analysis results (silent regions)
- Frequency domain analysis results (spectral features)
- Pitch and timbre analysis results (pitch statistics, MFCC statistics)
- Anomaly detection results (amplitude and spectral anomalies)

### Summary Report

A summary report is generated in both CSV and HTML formats, containing key metrics for all analyzed files.

## Examples

### Basic Analysis

```bash
python run_audio_eda.py --input data/test1 --output_dir results/eda
```

### Analysis with Preprocessing

```bash
python run_audio_eda.py --input data/test1 --output_dir results/eda --normalize --remove_silence
```

### Analysis with Advanced Preprocessing

```bash
python run_audio_eda.py --input data/test1 --output_dir results/eda --normalize --remove_silence --filter_type lowpass --cutoff_freq 4000 --fade_in 0.1 --fade_out 0.1
```

### Analysis with Bandpass Filter

```bash
python run_audio_eda.py --input data/test1 --output_dir results/eda --filter_type bandpass --cutoff_low 300 --cutoff_high 3000
```

## Interpreting Results

### Descriptive Statistics

- **Mean**: Average amplitude of the audio signal
- **Standard Deviation**: Variation in amplitude
- **RMS**: Root Mean Square value, related to the power of the signal
- **Zero Crossings**: Number of times the signal crosses the zero line, related to frequency content

### Time Domain Analysis

- **Waveform**: Visual representation of amplitude over time
- **Envelope**: Outline of the waveform, showing the overall shape
- **Energy**: Power of the signal over time
- **Zero Crossing Rate**: Frequency of sign changes, related to noisiness and pitch

### Frequency Domain Analysis

- **FFT Magnitude Spectrum**: Distribution of frequencies in the signal
- **Spectrogram**: Time-frequency representation showing how frequency content changes over time
- **Spectral Centroid**: Center of mass of the spectrum, related to brightness
- **Spectral Bandwidth**: Width of the spectrum, related to noisiness
- **Spectral Rolloff**: Frequency below which most of the energy is concentrated
- **Spectral Flatness**: Ratio of geometric mean to arithmetic mean of the spectrum, related to tonality

### Pitch and Timbre Analysis

- **Pitch**: Fundamental frequency of the audio
- **MFCC**: Mel-frequency cepstral coefficients, representing the timbre or sound quality

### Anomaly Detection

- **Amplitude Anomalies**: Unusual amplitude values
- **Spectral Anomalies**: Unusual frequency content

## Advanced Usage

### Custom Preprocessing Pipeline

You can create a custom preprocessing pipeline using the `preprocess_for_eda` function:

```python
from src.preprocessing.audio import preprocess_audio
from src.eda.preprocessing import preprocess_for_eda

# Load audio
audio_data, sample_rate = preprocess_audio("data/test1/test1_01.wav")

# Apply custom preprocessing
processed_audio = preprocess_for_eda(
    audio_data,
    sample_rate,
    normalize=True,
    remove_silence=True,
    trim_start=1.0,
    trim_end=5.0,
    fade_in=0.2,
    fade_out=0.2,
    filter_type="bandpass",
    cutoff_freq=(500, 4000)
)
```

### Batch Processing with Custom Parameters

```python
from src.utils.file_utils import get_audio_files
from src.eda.audio_eda import batch_analyze_audio_files

# Get all audio files
file_paths = get_audio_files("data/test1")

# Analyze with custom parameters
results = batch_analyze_audio_files(
    file_paths,
    output_dir="results/eda_custom",
    normalize=True,
    remove_silence=True,
    filter_type="highpass",
    cutoff_freq=300
)
```
