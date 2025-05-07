# CTC Speech Transcription

This project implements a CTC (Connectionist Temporal Classification) decoder for accurate speech transcription. It uses state-of-the-art pretrained models and provides options for different decoding strategies to maximize transcription accuracy.

## Project Structure

```
CTC-SpeechRefinement/
├── config/                 # Configuration files
├── data/                   # Audio data
│   ├── test1/              # Test set 1
│   └── test2/              # Test set 2
├── docs/                   # Documentation
├── models/                 # Saved models
├── results/                # Evaluation results
├── src/                    # Source code
│   ├── preprocessing/      # Audio preprocessing
│   ├── features/           # Feature extraction
│   ├── models/             # Acoustic models
│   ├── decoder/            # CTC decoder
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── transcripts/            # Generated transcriptions
├── requirements.txt        # Dependencies
├── transcribe.py           # Main script
└── README.md               # Project documentation
```

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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Transcription

To transcribe audio files using the default settings:

```bash
python transcribe.py --input_dir data/test1
```

This will:
1. Load audio files from the specified directory
2. Preprocess the audio
3. Run the audio through a pretrained acoustic model
4. Decode the model outputs using CTC decoding
5. Save the transcriptions to the `transcripts` directory

### Advanced Options

The script supports several options for customizing the transcription process:

```bash
python transcribe.py --input_dir data/test1 \
                    --output_dir transcripts \
                    --results_dir results \
                    --model_name facebook/wav2vec2-base-960h \
                    --decoder_type beam_search \
                    --beam_width 100 \
                    --normalize_audio \
                    --remove_silence
```

### Evaluation

If you have reference transcriptions, you can evaluate the quality of the generated transcriptions:

```bash
python transcribe.py --input_dir data/test1 \
                    --reference_dir reference_transcripts
```

This will compute Word Error Rate (WER) and Character Error Rate (CER) metrics and save the results to the `results` directory.

## Implementation Approach

### Audio Preprocessing

The preprocessing module handles:
- Loading audio files with the correct sample rate
- Normalizing audio to have zero mean and unit variance
- Removing silence to improve transcription accuracy

### Acoustic Model

We use pretrained models from the Hugging Face Transformers library, specifically the Wav2Vec2 model which has been trained on large amounts of speech data. This model converts audio into a sequence of logits representing the probability of each character at each timestep.

### CTC Decoding

The CTC decoder converts the model's output logits into text transcriptions. We support two decoding strategies:

1. **Greedy Decoding**: Simply takes the most likely character at each timestep and collapses repeated characters.
2. **Beam Search Decoding**: Maintains multiple hypotheses and uses a language model to improve transcription accuracy.

### Evaluation

We evaluate transcription quality using:
- Word Error Rate (WER): The percentage of words that are incorrectly transcribed
- Character Error Rate (CER): The percentage of characters that are incorrectly transcribed

## Results

The system achieves high transcription accuracy on the test1 dataset by:
1. Using a state-of-the-art pretrained acoustic model
2. Applying appropriate preprocessing to improve audio quality
3. Using beam search decoding with a language model to correct errors

## License

This project is licensed under the MIT License - see the LICENSE file for details.
