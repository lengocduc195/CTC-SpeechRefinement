# Speculative Decoding with CTC-Drafter and CR-CTC

This document explains the implementation of speculative decoding with CTC-Drafter and Consistency-Regularized CTC (CR-CTC) for speech recognition.

## Overview

Speculative decoding is a technique to speed up inference in sequence generation models by using a smaller, faster model (the "drafter") to propose candidate outputs that are then verified by a larger, more accurate model (the "verifier"). This approach can significantly reduce inference time while maintaining high accuracy.

Our implementation combines two key techniques:

1. **CTC-Drafter**: Uses a smaller, faster model to generate initial transcriptions that are then verified by a larger model.
2. **Consistency-Regularized CTC (CR-CTC)**: Enhances CTC decoding by enforcing consistency between different perturbations of the input, leading to more robust transcriptions.

## Architecture

The speculative decoding system consists of three main components:

### 1. CTC-Drafter

The CTC-Drafter uses a two-model approach:

- **Drafter Model**: A smaller, faster model (e.g., wav2vec2-base) that generates initial transcriptions.
- **Verifier Model**: A larger, more accurate model (e.g., wav2vec2-large) that verifies and corrects the draft transcriptions.

The process works as follows:

1. The drafter model generates a draft transcription.
2. The verifier model checks the draft and produces a verified transcription.
3. The system calculates an acceptance rate based on how much of the draft was accepted.

### 2. Consistency-Regularized CTC (CR-CTC)

CR-CTC enhances the robustness of transcriptions by:

1. Generating multiple perturbations of the input audio (e.g., time stretching, pitch shifting, adding noise).
2. Decoding each perturbation to get multiple transcription candidates.
3. Applying consistency voting to select the most consistent transcription.

This approach helps to filter out noise and artifacts in the transcription process, leading to more accurate results.

### 3. Speculative Decoder

The Speculative Decoder combines CTC-Drafter and CR-CTC:

1. CTC-Drafter generates an initial draft transcription.
2. The system decides whether to use the draft based on the acceptance rate.
3. If CR-CTC is enabled, it further refines the transcription by enforcing consistency.

## Implementation Details

### CTC-Drafter

The CTC-Drafter is implemented in `ctc_speech_refinement/core/decoder/ctc_drafter.py` and includes:

- `draft()`: Generates a draft transcription using the drafter model.
- `verify()`: Verifies and corrects the draft using the verifier model.
- `speculative_decode()`: Combines drafting and verification.

### CR-CTC

The CR-CTC is implemented in `ctc_speech_refinement/core/decoder/cr_ctc.py` and includes:

- `generate_perturbations()`: Creates perturbed versions of the input audio.
- `decode_perturbations()`: Decodes each perturbation to get transcriptions.
- `apply_consistency_voting()`: Selects the most consistent transcription.

### Speculative Decoder

The Speculative Decoder is implemented in `ctc_speech_refinement/core/decoder/speculative_decoder.py` and includes:

- `decode()`: Performs speculative decoding with CR-CTC verification.
- `batch_decode()`: Processes multiple audio files.

## Usage

### Command-Line Interface

The easiest way to use speculative decoding is through the command-line interface:

```bash
python run_speculative_decoding.py --input_dir data/test1 --output_dir transcripts --results_dir results
```

#### Basic Options

- `--input_dir`: Directory containing audio files to transcribe
- `--output_dir`: Directory to save transcriptions
- `--results_dir`: Directory to save results

#### Model Options

- `--drafter_model`: Pretrained model name or path for the drafter (default: "facebook/wav2vec2-base-960h")
- `--verifier_model`: Pretrained model name or path for the verifier (default: "facebook/wav2vec2-large-960h-lv60-self")

#### Decoder Options

- `--decoder_type`: Type of CTC decoder to use (choices: "greedy", "beam_search")
- `--beam_width`: Beam width for beam search decoding

#### Speculative Decoding Options

- `--max_draft_length`: Maximum length of draft sequences
- `--draft_timeout_ms`: Maximum time in milliseconds to spend on drafting
- `--acceptance_threshold`: Threshold for acceptance rate to determine whether to use the draft

#### CR-CTC Options

- `--use_cr_ctc`: Whether to use CR-CTC for verification
- `--num_perturbations`: Number of perturbations to generate for CR-CTC
- `--fallback_to_standard`: Whether to fall back to standard decoding if acceptance rate is below threshold

#### Audio Preprocessing Options

- `--normalize_audio`: Normalize audio data
- `--remove_silence`: Remove silent regions from audio
- `--apply_vad`: Apply Voice Activity Detection
- `--vad_method`: VAD method to use (choices: "energy", "zcr")
- `--reduce_noise`: Apply noise reduction
- `--noise_reduction_method`: Noise reduction method to use
- `--normalize_frequency`: Apply frequency normalization
- `--frequency_normalization_method`: Frequency normalization method to use

#### Evaluation Options

- `--reference_dir`: Directory containing reference transcriptions for evaluation

### Python API

You can also use the speculative decoding system directly in your Python code:

```python
from ctc_speech_refinement.core.models.acoustic_model import AcousticModel
from ctc_speech_refinement.core.decoder.ctc_decoder import CTCDecoder
from ctc_speech_refinement.core.decoder.ctc_drafter import CTCDrafter
from ctc_speech_refinement.core.decoder.cr_ctc import CRCTC
from ctc_speech_refinement.core.decoder.speculative_decoder import SpeculativeDecoder

# Initialize models and decoders
drafter_model = AcousticModel(model_name="facebook/wav2vec2-base-960h")
verifier_model = AcousticModel(model_name="facebook/wav2vec2-large-960h-lv60-self")

drafter_decoder = CTCDecoder(processor=drafter_model.processor, decoder_type="beam_search")
verifier_decoder = CTCDecoder(processor=verifier_model.processor, decoder_type="beam_search")

# Initialize CTC-Drafter
drafter = CTCDrafter(
    drafter_model=drafter_model,
    verifier_model=verifier_model,
    drafter_decoder=drafter_decoder,
    verifier_decoder=verifier_decoder
)

# Initialize CR-CTC
cr_ctc = CRCTC(
    model=verifier_model,
    decoder=verifier_decoder,
    num_perturbations=3
)

# Initialize Speculative Decoder
speculative_decoder = SpeculativeDecoder(
    drafter=drafter,
    verifier=cr_ctc,
    use_cr_ctc_for_verification=True,
    fallback_to_standard_decoding=True,
    acceptance_threshold=0.5
)

# Perform speculative decoding
results = speculative_decoder.decode(audio_data, sample_rate)
```

## Performance Considerations

### Speed vs. Accuracy Trade-offs

- **Drafter Model Size**: Smaller models are faster but may produce less accurate drafts.
- **Verifier Model Size**: Larger models are more accurate but slower.
- **Number of Perturbations**: More perturbations in CR-CTC improve robustness but increase computation time.
- **Acceptance Threshold**: Higher thresholds lead to more fallbacks to standard decoding, which is slower but potentially more accurate.

### Optimizations

- **Batch Processing**: Process multiple audio files in parallel for better throughput.
- **Model Quantization**: Use quantized models for faster inference.
- **GPU Acceleration**: Use GPU for model inference when available.

## Results and Evaluation

The system generates detailed results including:

- Transcriptions for each audio file
- Timing information (draft time, verification time, total time)
- Acceptance rates for drafts
- Consistency metrics for CR-CTC

If reference transcriptions are provided, the system also calculates:

- Word Error Rate (WER)
- Character Error Rate (CER)

## References

1. Speculative Decoding: https://arxiv.org/abs/2211.17192
2. CTC-based models: https://distill.pub/2017/ctc/
3. Consistency Training: https://arxiv.org/abs/2010.07079
