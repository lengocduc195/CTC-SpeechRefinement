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

- **Comprehensive evaluation**:
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Syllable Error Rate (SER)
  - Decoding latency measurements
  - Detailed reports and visualizations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vietnamese-asr-ctc-evaluation.git
   cd vietnamese-asr-ctc-evaluation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install KenLM for language model support:
   ```bash
   pip install https://github.com/kpu/kenlm/archive/master.zip
   ```

## Usage

### Basic Usage

Run the evaluation with default settings (all tokenizers):

```bash
python ctc_eval.py
```

This will:
1. Create sample data if none exists
2. Initialize tokenizers for all levels
3. Evaluate CTC decoding performance
4. Generate a comprehensive report

### Tokenizer Options

You can specify which tokenizers to use:

```bash
# Use only character and syllable tokenizers
python ctc_eval.py --tokenizers character,syllable

# Use only word tokenizer
python ctc_eval.py --tokenizers word

# Use all tokenizers (default)
python ctc_eval.py --tokenizers all
```

### Customizing Tokenizers

You can customize each tokenizer with additional options:

```bash
# Customize subword tokenizer
python ctc_eval.py --tokenizers subword --subword_vocab_size 10000 --subword_model_path data/tokenizers/vietnamese_bpe_10k.model

# Customize syllable and word tokenizers
python ctc_eval.py --tokenizers syllable,word --syllable_vocab_path custom_syllables.txt --word_vocab_path custom_words.txt
```

### Complete Command-line Options

```bash
python ctc_eval.py [OPTIONS]
```

#### Basic Options:
- `--audio_dir`: Directory containing audio files (default: "data/audio")
- `--transcript_dir`: Directory containing transcript files (default: "data/transcripts")
- `--output_dir`: Directory to save results (default: "results")
- `--num_samples`: Number of samples to evaluate (default: 10)

#### Decoding Options:
- `--beam_size`: Beam size for beam search decoding (default: 10)
- `--alpha`: Language model weight (default: 0.5)
- `--beta`: Length penalty (default: 1.0)

#### Tokenizer Options:
- `--tokenizers`: Comma-separated list of tokenizers to use: 'character', 'subword', 'syllable', 'word', or 'all' (default: "all")
- `--subword_vocab_size`: Vocabulary size for subword tokenizer (default: 5000)
- `--subword_model_path`: Path to SentencePiece model for subword tokenization (default: "data/tokenizers/vietnamese_bpe_5000.model")
- `--syllable_vocab_path`: Path to syllable vocabulary file (default: "data/tokenizers/vietnamese_syllables.txt")
- `--word_vocab_path`: Path to word vocabulary file (default: "data/tokenizers/vietnamese_words.txt")

### Examples

1. **Evaluate only character-level tokenization**:
   ```bash
   python ctc_eval.py --tokenizers character
   ```

2. **Compare syllable and word tokenization**:
   ```bash
   python ctc_eval.py --tokenizers syllable,word
   ```

3. **Evaluate subword tokenization with custom vocabulary size**:
   ```bash
   python ctc_eval.py --tokenizers subword --subword_vocab_size 8000
   ```

4. **Use custom vocabulary files**:
   ```bash
   python ctc_eval.py --tokenizers syllable,word --syllable_vocab_path data/custom/vn_syllables.txt --word_vocab_path data/custom/vn_words.txt
   ```

5. **Adjust beam search parameters**:
   ```bash
   python ctc_eval.py --beam_size 20 --alpha 0.7 --beta 1.5
   ```

### Using Your Own Data

1. Place your audio files (WAV format) in the `data/audio` directory
2. Place corresponding transcripts in the `data/transcripts` directory
   - Transcript files should have the same base name as the audio files with a `.txt` extension
   - Example: `data/audio/sample_01.wav` → `data/transcripts/sample_01.txt`

3. (Optional) Add your own language models in ARPA format:
   - Character-level: `data/lm/character_lm.arpa`
   - Subword-level: `data/lm/subword_lm.arpa`
   - Syllable-level: `data/lm/syllable_lm.arpa`
   - Word-level: `data/lm/word_lm.arpa`

4. (Optional) Train your own BPE model:
   ```python
   from ctc_eval import SubwordTokenizer
   tokenizer = SubwordTokenizer(vocab_size=5000)
   tokenizer.train('data/train.txt', 'data/tokenizers/vietnamese_bpe_5000')
   ```

## Creating Custom Vocabulary Files

### Syllable Vocabulary

Create a text file with one Vietnamese syllable per line:

```
xin
chào
tôi
là
người
việt
nam
...
```

### Word Vocabulary

Create a text file with one Vietnamese word per line (can be multi-syllable):

```
xin chào
cảm ơn
việt nam
học sinh
...
```

## Understanding CTC Decoding

### CTC Basics

Connectionist Temporal Classification (CTC) is a technique for sequence-to-sequence learning without requiring aligned data. Key concepts:

- **Blank token (`<blank>`)**: Represents "no output" and helps with alignment
- **CTC path collapse**: Removing repeated tokens and blank tokens to get the final output
- **Beam search**: Exploring multiple hypotheses to find the most likely sequence

### Tokenization Levels

- **Character-level**:
  - Pros: Smallest vocabulary, no OOV issues
  - Cons: Longer sequences, slower decoding

- **Subword-level (BPE)**:
  - Pros: Good balance between vocabulary size and sequence length
  - Cons: May produce suboptimal segmentation for Vietnamese

- **Syllable-level**:
  - Pros: Natural unit for Vietnamese, shorter sequences than characters
  - Cons: Larger vocabulary than characters, potential OOV issues

- **Word-level**:
  - Pros: Shortest sequences, fastest decoding
  - Cons: Largest vocabulary, significant OOV issues

## Output

The evaluation generates:

1. **Summary CSV**: Basic metrics for each tokenization level
2. **Visualizations**:
   - Error rates comparison
   - Decoding latency comparison
   - Accuracy vs. speed trade-off
3. **HTML Report**: Comprehensive analysis with visualizations
4. **Markdown Report**: Same content in Markdown format

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses PyTorch for tensor operations
- SentencePiece for BPE tokenization
- KenLM for language model integration
- JiWER for WER/CER calculation
