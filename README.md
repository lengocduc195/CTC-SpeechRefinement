# Vietnamese ASR CTC Decoder Evaluation

This project evaluates different tokenization levels for Vietnamese Automatic Speech Recognition (ASR) using Connectionist Temporal Classification (CTC) decoding. It compares character-level, subword-level, syllable-level, and word-level tokenization in terms of accuracy and decoding speed.

## Features

- **Multiple tokenization levels**:
  - Character-level: All Vietnamese characters + diacritics
  - Subword-level: Byte-Pair Encoding (BPE) with configurable vocabulary size
  - Syllable-level: Vietnamese syllables as tokens
  - Word-level: Vietnamese words with OOV fallback to subwords

- **CTC decoding algorithms**:
  - Greedy decoding
  - Beam search with language model integration
  - Speculative decoding with CTC-Drafter and CR-CTC

- **Comprehensive evaluation**:
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Syllable Error Rate (SER)
  - Decoding latency measurements
  - Detailed reports and visualizations

- **Audio EDA (Exploratory Data Analysis)**:
  - Time-domain analysis
  - Frequency-domain analysis
  - Amplitude/energy analysis
  - Pitch/timbre analysis
  - Anomaly detection
  - Data preprocessing options

## Project Structure

```
CTC-SpeechRefinement/
├── ctc_speech_refinement/    # Main package
│   ├── apps/                 # Application modules
│   │   ├── audio_eda/        # Audio EDA application
│   │   ├── speculative_decoding/ # Speculative decoding application
│   │   ├── transcription/    # Transcription application
│   │   └── ui/               # UI applications
│   ├── config/               # Configuration files
│   ├── core/                 # Core functionality
│   │   ├── decoder/          # CTC decoders
│   │   ├── eda/              # EDA functionality
│   │   ├── features/         # Feature extraction
│   │   ├── models/           # Acoustic models
│   │   ├── preprocessing/    # Audio preprocessing
│   │   ├── ui/               # UI components
│   │   └── utils/            # Utility functions
│   ├── docs/                 # Documentation
│   ├── tests/                # Unit tests
│   └── transcripts/          # Generated transcriptions
├── data/                     # Audio data and tokenizers
│   ├── audio/                # Audio files
│   ├── tokenizers/           # Tokenizer files
│   └── transcripts/          # Reference transcripts
├── docs/                     # Project documentation
├── notebooks/                # Jupyter notebooks for EDA
├── results/                  # Evaluation results
├── tests/                    # Project-level tests
├── transcripts/              # Generated transcriptions
├── requirements.txt          # Dependencies
├── run_audio_eda.py          # Script to run audio EDA
├── run_error_analysis.py     # Script to run error analysis
├── run_preprocessing_ui.py   # Script to run preprocessing UI
├── run_speculative_decoding.py # Script to run speculative decoding
├── run_transcription.py      # Script to run transcription
└── README.md                 # Project documentation
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

4. (Optional) Install KenLM for language model support:
   ```bash
   pip install https://github.com/kpu/kenlm/archive/master.zip
   ```

## Usage

### Speech Transcription

Run the transcription with default settings:

```bash
python run_transcription.py
```

With custom options:

```bash
python run_transcription.py --input_dir data/audio \
                           --output_dir transcripts \
                           --results_dir results \
                           --model_name facebook/wav2vec2-base-960h \
                           --decoder_type beam_search \
                           --beam_width 100 \
                           --normalize_audio \
                           --remove_silence \
                           --reference_dir data/transcripts
```

### Audio EDA (Exploratory Data Analysis)

Run audio EDA with default settings:

```bash
python run_audio_eda.py
```

With custom options:

```bash
python run_audio_eda.py --input_dir data/audio \
                       --output_dir results/eda \
                       --analysis_types time,frequency,pitch \
                       --visualize
```

### Speculative Decoding

Run speculative decoding with default settings:

```bash
python run_speculative_decoding.py
```

With custom options:

```bash
python run_speculative_decoding.py --input_dir data/audio \
                                  --output_dir transcripts/speculative \
                                  --results_dir results/speculative \
                                  --model_name facebook/wav2vec2-base-960h \
                                  --drafter_type ctc_drafter \
                                  --acceptance_threshold 0.8
```

### Error Analysis

Run error analysis on transcription results:

```bash
python run_error_analysis.py --reference_dir data/transcripts \
                            --hypothesis_dir transcripts \
                            --output_dir results/error_analysis
```

### Preprocessing UI

Launch the audio preprocessing UI:

```bash
python run_preprocessing_ui.py
```

## Jupyter Notebooks

The project includes several Jupyter notebooks for interactive audio analysis:

1. `01_Basic_Audio_EDA.ipynb`: Basic audio exploration
2. `02_Audio_Preprocessing.ipynb`: Audio preprocessing techniques
3. `03_Frequency_Domain_Analysis.ipynb`: Frequency analysis
4. `04_Pitch_Timbre_Analysis.ipynb`: Pitch and timbre analysis
5. `05_Anomaly_Detection.ipynb`: Audio anomaly detection
6. `06_Batch_Audio_Analysis.ipynb`: Batch processing of audio files
7. `07_Audio_Visualization_Techniques.ipynb`: Advanced visualization techniques

To run the notebooks:

```bash
jupyter notebook notebooks/
```

## Documentation

For more detailed information, refer to the documentation in the `docs/` directory:

- `ARCHITECTURE.md`: System architecture overview
- `AUDIO_EDA_GUIDE.md`: Guide to audio EDA features
- `AUDIO_PREPROCESSING_GUIDE.md`: Guide to audio preprocessing
- `DEVELOPER_GUIDE.md`: Guide for developers
- `ERROR_ANALYSIS_GUIDE.md`: Guide to error analysis
- `INTEGRATION_GUIDE.md`: Guide for integrating with other systems
- `SPECULATIVE_DECODING.md`: Guide to speculative decoding
- `TECHNICAL_DEBT.md`: Known limitations and technical debt
- `USAGE_EXAMPLES.md`: Additional usage examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses PyTorch for tensor operations
- SentencePiece for BPE tokenization
- KenLM for language model integration
- JiWER for WER/CER calculation
- Transformers library for pretrained models
