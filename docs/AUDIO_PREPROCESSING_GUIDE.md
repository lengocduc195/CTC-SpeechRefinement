# Audio Preprocessing Guide

This guide explains the audio preprocessing features available in the CTC-SpeechRefinement project.

## Overview

The audio preprocessing module provides various techniques to improve the quality of audio data before speech recognition. These techniques include:

1. **Amplitude Normalization**: Normalizes the amplitude of the audio signal.
2. **Silence Removal**: Removes silent segments from the audio.
3. **Voice Activity Detection (VAD)**: Detects and extracts speech segments.
4. **Noise Reduction**: Reduces background noise using various methods.
5. **Frequency Normalization**: Normalizes the frequency content of the audio.

## Using the Preprocessing UI

The preprocessing UI provides a graphical interface for configuring and applying preprocessing options to audio files.

### Running the UI

To run the preprocessing UI, use the following command:

```bash
python run_preprocessing_ui.py --language vi
```

Options:
- `--language` or `-l`: UI language (en or vi). Default is "vi" (Vietnamese).

### UI Features

The UI includes the following sections:

1. **Input/Output**: Select input audio file(s) and output directory.
2. **Preprocessing Options**: Configure preprocessing options.
3. **Preview**: Preview the effect of preprocessing on a selected audio file.
4. **Configuration**: Save and load preprocessing configurations.

## Preprocessing Options

### Amplitude Normalization (Chuẩn hóa biên độ)

Normalizes the audio data to have zero mean and unit variance, which helps to standardize the volume level across different recordings.

### Silence Removal (Loại bỏ khoảng lặng)

Removes silent segments from the audio, which can improve speech recognition accuracy and reduce processing time.

### Voice Activity Detection (VAD)

Detects and extracts speech segments from the audio, ignoring non-speech segments.

Available methods:
- **Energy-based (Dựa trên năng lượng)**: Detects speech based on the energy level of the audio.
- **Zero-crossing rate (Tỷ lệ giao điểm không)**: Detects speech based on the zero-crossing rate and energy level.

### Noise Reduction (Khử nhiễu)

Reduces background noise in the audio, which can improve speech recognition accuracy.

Available methods:
- **Spectral Subtraction (Trừ phổ)**: Estimates the noise spectrum and subtracts it from the audio spectrum.
- **Wiener Filter (Bộ lọc Wiener)**: Applies a Wiener filter to reduce noise.
- **Median Filter (Bộ lọc trung vị)**: Applies a median filter to reduce impulsive noise.
- **Noise Reduce Library (Thư viện khử nhiễu)**: Uses the noisereduce library for noise reduction.

### Frequency Normalization (Chuẩn hóa tần số)

Normalizes the frequency content of the audio, which can improve speech recognition accuracy.

Available methods:
- **Bandpass Filter (Bộ lọc thông dải)**: Applies a bandpass filter to focus on the speech frequency range.
- **Pre-emphasis (Nhấn mạnh trước)**: Boosts high frequencies to emphasize speech.
- **Spectral Equalization (Cân bằng phổ)**: Flattens the frequency spectrum.
- **Combined (Kết hợp)**: Applies a combination of the above methods.

## Using the API

You can also use the preprocessing API directly in your Python code:

```python
from src.preprocessing.audio import preprocess_audio, batch_preprocess

# Preprocess a single audio file
audio_data, sample_rate = preprocess_audio(
    "data/test1/test1_01.wav",
    normalize=True,
    remove_silence_flag=True,
    apply_vad_flag=True,
    vad_method="energy",
    reduce_noise_flag=True,
    noise_reduction_method="spectral_subtraction",
    normalize_frequency_flag=True,
    frequency_normalization_method="bandpass"
)

# Preprocess multiple audio files
results = batch_preprocess(
    ["data/test1/test1_01.wav", "data/test1/test1_02.wav"],
    output_dir="data/preprocessed",
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

## Advanced Usage

### Voice Activity Detection (VAD)

You can use the VAD module directly:

```python
from src.preprocessing.vad import apply_vad, energy_vad, zcr_vad

# Load audio
audio_data, sample_rate = librosa.load("data/test1/test1_01.wav", sr=16000)

# Apply VAD
speech_audio = apply_vad(audio_data, sample_rate, method="energy")

# Get speech regions
speech_regions = energy_vad(audio_data, sample_rate)
for start_time, end_time in speech_regions:
    print(f"Speech from {start_time:.2f}s to {end_time:.2f}s")
```

### Noise Reduction

You can use the noise reduction module directly:

```python
from src.preprocessing.noise_reduction import reduce_noise, spectral_subtraction, wiener_filter

# Load audio
audio_data, sample_rate = librosa.load("data/test1/test1_01.wav", sr=16000)

# Apply noise reduction
denoised_audio = reduce_noise(audio_data, sample_rate, method="spectral_subtraction")

# Use specific method
denoised_audio = spectral_subtraction(audio_data, sample_rate)
```

### Frequency Normalization

You can use the frequency normalization module directly:

```python
from src.preprocessing.frequency_normalization import normalize_frequency, apply_bandpass_filter, apply_preemphasis

# Load audio
audio_data, sample_rate = librosa.load("data/test1/test1_01.wav", sr=16000)

# Apply frequency normalization
normalized_audio = normalize_frequency(audio_data, sample_rate, method="bandpass")

# Use specific method
filtered_audio = apply_bandpass_filter(audio_data, sample_rate, low_freq=80.0, high_freq=8000.0)
```

## Best Practices

For optimal speech recognition results, consider the following preprocessing pipeline:

1. **Frequency Normalization**: Apply bandpass filtering to focus on the speech frequency range (80-8000 Hz).
2. **Noise Reduction**: Apply spectral subtraction or Wiener filtering to reduce background noise.
3. **Voice Activity Detection**: Apply VAD to extract speech segments.
4. **Amplitude Normalization**: Normalize the amplitude of the speech segments.

This pipeline helps to improve the quality of the audio data and can significantly improve speech recognition accuracy.

## Troubleshooting

### Common Issues

1. **No speech detected**: Try adjusting the VAD parameters or using a different VAD method.
2. **Excessive noise reduction**: Try a different noise reduction method or adjust the parameters.
3. **Distorted audio**: Check if the frequency normalization parameters are appropriate for your audio.

### Getting Help

If you encounter any issues with the preprocessing module, please check the logs for error messages and refer to the API documentation for more information.

## References

- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Noise Reduce Library](https://github.com/timsainb/noisereduce)
