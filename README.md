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
├── notebooks/                # Jupyter notebooks for EDA
├── requirements.txt          # Dependencies
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
python run.py
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses PyTorch for tensor operations
- SentencePiece for BPE tokenization
- KenLM for language model integration
- JiWER for WER/CER calculation
- Transformers library for pretrained models
