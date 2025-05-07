# CTC Speech Refinement

A speech recognition system using CTC decoding with speculative decoding and consistency regularization.

## Project Structure

The project has been restructured to follow best practices for Python packages:

```
CTC-SpeechRefinement/
├── ctc_speech_refinement/     # Main package
│   ├── core/                  # Core functionality
│   │   ├── preprocessing/     # Audio preprocessing
│   │   ├── features/          # Feature extraction
│   │   ├── models/            # Acoustic models
│   │   ├── decoder/           # CTC decoder implementations
│   │   ├── utils/             # Utility functions
│   │   └── eda/               # Exploratory data analysis
│   ├── apps/                  # Applications
│   │   ├── transcription/     # Basic transcription
│   │   ├── speculative_decoding/ # Speculative decoding
│   │   ├── audio_eda/         # Audio EDA tools
│   │   └── ui/                # User interfaces
│   ├── config/                # Configuration
│   ├── tests/                 # Unit tests
│   └── setup.py               # Package setup
├── data/                      # Audio data
│   ├── test1/                 # Test set 1
│   └── test2/                 # Test set 2
├── docs/                      # Documentation
├── models/                    # Saved models
├── results/                   # Evaluation results
├── transcripts/               # Generated transcriptions
├── run_transcription.py       # Script to run transcription
├── run_speculative_decoding.py # Script to run speculative decoding
├── run_audio_eda_new.py       # Script to run audio EDA
├── run_preprocessing_ui_new.py # Script to run preprocessing UI
└── migrate_to_new_structure.py # Script to migrate code to new structure
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

3. Install the package in development mode:
```bash
pip install -e ctc_speech_refinement
```

## Migration to New Structure

To migrate the existing code to the new structure, run:

```bash
python migrate_to_new_structure.py
```

This will copy and update the files from the old structure to the new structure.

## Usage

### Basic Transcription

```bash
python run_transcription.py --input_dir data/test1 --output_dir transcripts
```

### Speculative Decoding

```bash
python run_speculative_decoding.py --input_dir data/test1 --output_dir transcripts --results_dir results
```

### Audio EDA

```bash
python run_audio_eda_new.py --input data/test1 --output_dir results/eda
```

### Preprocessing UI

```bash
python run_preprocessing_ui_new.py --language vi
```

## Documentation

For more detailed documentation, see the following guides:

- [Architecture Guide](docs/ARCHITECTURE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [Audio EDA Guide](docs/AUDIO_EDA_GUIDE.md)
- [Speculative Decoding Guide](docs/SPECULATIVE_DECODING.md)
- [Integration Guide](docs/INTEGRATION_GUIDE.md)
- [Audio Preprocessing Guide](docs/AUDIO_PREPROCESSING_GUIDE.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
