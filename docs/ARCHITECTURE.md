# CTC Speech Transcription Architecture

This document describes the high-level architecture of the CTC Speech Transcription system.

## System Overview

The system is designed to transcribe speech audio files using Connectionist Temporal Classification (CTC) decoding. It follows a modular architecture with clear separation of concerns:

```
[Audio Files] → [Preprocessing] → [Acoustic Model] → [CTC Decoder] → [Transcriptions]
```

## Key Components

### 1. Audio Preprocessing (`src/preprocessing/`)

Responsible for preparing audio data for the model:

- **Loading**: Reads audio files and resamples to the target sample rate (16kHz)
- **Normalization**: Scales audio to have zero mean and unit variance
- **Silence Removal**: Detects and removes silent segments to improve transcription

### 2. Feature Extraction (`src/features/`)

Extracts acoustic features from the preprocessed audio:

- **MFCC**: Mel-frequency cepstral coefficients
- **Mel Spectrograms**: Time-frequency representation using mel scale
- **Spectrograms**: Standard time-frequency representation

### 3. Acoustic Model (`src/models/`)

Converts audio features into character probabilities:

- Uses pretrained Wav2Vec2 model from Hugging Face
- Outputs logits representing character probabilities at each timestep
- Handles batching and device management (CPU/GPU)

### 4. CTC Decoder (`src/decoder/`)

Converts model outputs into text transcriptions:

- **Greedy Decoding**: Takes most likely character at each timestep
- **Beam Search Decoding**: Maintains multiple hypotheses and uses language model
- Handles blank tokens and repeated characters according to CTC algorithm

### 5. Utilities (`src/utils/`)

Supporting functionality:

- **File Handling**: Loading/saving transcriptions and results
- **Evaluation**: Computing WER and CER metrics
- **Visualization**: Plotting results

## Data Flow

1. Audio files are loaded and preprocessed
2. Preprocessed audio is passed to the acoustic model
3. Model outputs logits representing character probabilities
4. CTC decoder converts logits to text transcriptions
5. Transcriptions are saved and optionally evaluated against references

## Configuration

The system is configured through the `config/config.py` file, which includes settings for:

- Data directories
- Audio preprocessing parameters
- Feature extraction settings
- Model selection
- Decoder parameters

## Design Principles

1. **Modularity**: Each component has a clear responsibility and can be developed/tested independently
2. **Configurability**: Key parameters can be adjusted without code changes
3. **Extensibility**: New models, decoders, or preprocessing steps can be added easily
4. **Robustness**: Comprehensive error handling and logging

## Performance Considerations

- Batch processing is used to improve throughput
- GPU acceleration is used when available
- Preprocessing steps can be enabled/disabled based on requirements
- Beam search parameters can be tuned for accuracy vs. speed tradeoffs
