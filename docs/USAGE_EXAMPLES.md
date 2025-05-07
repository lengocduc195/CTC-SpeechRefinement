# Usage Examples

This document provides examples of how to use the CTC Speech Transcription system for various tasks.

## Basic Transcription

### Transcribing a Single Directory of Audio Files

```bash
python transcribe.py --input_dir data/test1
```

This will:
1. Load all audio files from the `data/test1` directory
2. Transcribe them using the default settings
3. Save the transcriptions to the `transcripts` directory

### Transcribing with Custom Output Directory

```bash
python transcribe.py --input_dir data/test1 --output_dir my_transcripts
```

This will save the transcriptions to the `my_transcripts` directory.

## Advanced Transcription Options

### Using Beam Search Decoding

```bash
python transcribe.py --input_dir data/test1 --decoder_type beam_search --beam_width 100
```

This uses beam search decoding with a beam width of 100, which generally produces more accurate transcriptions than greedy decoding.

### Using a Different Pretrained Model

```bash
python transcribe.py --input_dir data/test1 --model_name facebook/wav2vec2-large-960h-lv60-self
```

This uses a larger, more accurate pretrained model.

### Enabling Audio Preprocessing

```bash
python transcribe.py --input_dir data/test1 --normalize_audio --remove_silence
```

This enables audio normalization and silence removal, which can improve transcription accuracy for noisy audio.

## Evaluation

### Evaluating Against Reference Transcriptions

```bash
python transcribe.py --input_dir data/test1 --reference_dir reference_transcripts
```

This will:
1. Transcribe the audio files
2. Load reference transcriptions from the `reference_transcripts` directory
3. Compute WER and CER metrics
4. Save the evaluation results to the `results` directory

## Batch Processing

### Processing Multiple Directories

```bash
for dir in data/test1 data/test2; do
    python transcribe.py --input_dir $dir --output_dir transcripts/$(basename $dir)
done
```

This processes each directory separately and saves the transcriptions to separate subdirectories.

## Using as a Module

You can also use the components of the system in your own Python code:

```python
from src.preprocessing.audio import preprocess_audio
from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder

# Preprocess audio
audio_data, sample_rate = preprocess_audio("path/to/audio.wav")

# Initialize model and decoder
model = AcousticModel()
decoder = CTCDecoder(model.processor)

# Get logits from model
logits = model.get_logits(audio_data, sample_rate)

# Decode to get transcription
transcription = decoder.decode(logits)

print(transcription)
```

## Custom Pipeline Example

This example shows how to create a custom pipeline that uses different preprocessing steps:

```python
import os
from src.preprocessing.audio import load_audio, normalize_audio
from src.features.extraction import extract_mel_spectrogram, normalize_features
from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder
from src.utils.file_utils import save_transcription

# Load and preprocess audio
audio_file = "data/test1/test1_01.wav"
audio_data, sample_rate = load_audio(audio_file)
audio_data = normalize_audio(audio_data)

# Extract features
features = extract_mel_spectrogram(audio_data, sample_rate)
features = normalize_features(features)

# Initialize model and decoder
model = AcousticModel()
decoder = CTCDecoder(model.processor, decoder_type="beam_search", beam_width=100)

# Get logits from model
logits = model.get_logits(audio_data, sample_rate)

# Decode to get transcription
transcription = decoder.decode(logits)

# Save transcription
output_file = save_transcription(transcription, audio_file, "my_transcripts")
print(f"Transcription saved to {output_file}")
```

## Real-time Transcription Example

This example shows how to set up a simple real-time transcription system:

```python
import sounddevice as sd
import numpy as np
from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder

# Parameters
sample_rate = 16000
duration = 5  # seconds

# Initialize model and decoder
model = AcousticModel()
decoder = CTCDecoder(model.processor)

def callback(indata, frames, time, status):
    """This is called for each audio block."""
    if status:
        print(status)
    
    # Convert to mono if needed
    if indata.shape[1] > 1:
        audio_data = np.mean(indata, axis=1)
    else:
        audio_data = indata.flatten()
    
    # Get logits from model
    logits = model.get_logits(audio_data, sample_rate)
    
    # Decode to get transcription
    transcription = decoder.decode(logits)
    
    print(f"Transcription: {transcription}")

# Start recording
with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
    print(f"Recording for {duration} seconds...")
    sd.sleep(duration * 1000)
```

Note: This example requires the `sounddevice` package, which can be installed with `pip install sounddevice`.
