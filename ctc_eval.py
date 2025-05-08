#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CTC Decoder Evaluation for Vietnamese ASR
-----------------------------------------
This script evaluates different tokenization levels for Vietnamese ASR using CTC decoding.
It compares character-level, subword-level, syllable-level, and word-level tokenization
in terms of accuracy (CER, WER, SER) and decoding speed.

Author: AI Assistant
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict
import sentencepiece as spm
from tqdm import tqdm
import pandas as pd
import seaborn as sns

# Try to use jiwer for WER/CER calculation, but provide fallback implementations
try:
    from jiwer import wer as jiwer_wer
    from jiwer import cer as jiwer_cer
    JIWER_AVAILABLE = True
except ImportError:
    print("jiwer not available. Using custom WER/CER implementation.")
    JIWER_AVAILABLE = False

# Try to import KenLM for language model integration
try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    print("KenLM not available. Beam search with LM will not work.")
    KENLM_AVAILABLE = False

# Custom implementations of WER and CER for fallback
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def custom_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate.

    Args:
        reference: Reference text
        hypothesis: Hypothesis text

    Returns:
        Word Error Rate
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    distance = levenshtein_distance(' '.join(ref_words), ' '.join(hyp_words))
    return distance / max(len(ref_words), 1)

def custom_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate.

    Args:
        reference: Reference text
        hypothesis: Hypothesis text

    Returns:
        Character Error Rate
    """
    distance = levenshtein_distance(reference, hypothesis)
    return distance / max(len(reference), 1)

# Use jiwer if available, otherwise use custom implementations
def wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate."""
    if JIWER_AVAILABLE:
        try:
            return jiwer_wer(reference, hypothesis)
        except Exception:
            return custom_wer(reference, hypothesis)
    else:
        return custom_wer(reference, hypothesis)

def cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate."""
    if JIWER_AVAILABLE:
        try:
            return jiwer_cer(reference, hypothesis)
        except Exception:
            return custom_cer(reference, hypothesis)
    else:
        return custom_cer(reference, hypothesis)

# Constants
BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

@dataclass
class DecodingResult:
    """Class to store decoding results and metrics."""
    text: str
    tokens: List[str]
    decoding_time: float
    cer: Optional[float] = None
    wer: Optional[float] = None
    ser: Optional[float] = None

class TokenizerBase:
    """Base class for all tokenizers."""

    def __init__(self, name: str, vocab_size: int):
        self.name = name
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.blank_id = -1

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs to text."""
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        """Return vocabulary size including blank token."""
        return self.vocab_size + 1  # +1 for blank token

    def get_blank_id(self) -> int:
        """Return the ID of the blank token."""
        # Make sure blank_id is within the valid range
        vocab_size = self.get_vocab_size()
        if self.blank_id >= vocab_size:
            # If blank_id is out of bounds, use the last valid token ID
            return vocab_size - 1
        return self.blank_id

    def save(self, path: str) -> None:
        """Save tokenizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'name': self.name,
                'vocab_size': self.vocab_size,
                'id_to_token': self.id_to_token,
                'blank_id': self.blank_id
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TokenizerBase':
        """Load tokenizer from disk."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(data['name'], data['vocab_size'])
        tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        tokenizer.token_to_id = {v: int(k) for k, v in data['id_to_token'].items()}
        tokenizer.blank_id = data['blank_id']
        return tokenizer

class CharacterTokenizer(TokenizerBase):
    """Character-level tokenizer for Vietnamese."""

    def __init__(self, name: str = "character", vocab_size: int = 0):
        super().__init__(name, vocab_size)

        # Vietnamese characters (lowercase)
        base_chars = "abcdefghijklmnopqrstuvwxyz"
        # Vietnamese diacritics
        vn_chars = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
        # Digits and special characters
        digits = "0123456789"
        special = " ,.!?-:;\"'()[]{}%$#@&*+=/<>|~^_"

        # Combine all characters
        all_chars = base_chars + vn_chars + digits + special

        # Create vocabulary
        self.id_to_token = {i: c for i, c in enumerate(all_chars)}
        self.token_to_id = {c: i for i, c in enumerate(all_chars)}
        self.vocab_size = len(self.id_to_token)

        # Add blank token at the end
        self.blank_id = self.vocab_size
        self.id_to_token[self.blank_id] = BLANK_TOKEN
        self.token_to_id[BLANK_TOKEN] = self.blank_id

    def encode(self, text: str) -> List[int]:
        """Convert text to character IDs."""
        return [self.token_to_id.get(c, self.token_to_id.get(UNK_TOKEN, 0)) for c in text.lower()]

    def decode(self, ids: List[int]) -> str:
        """Convert character IDs to text."""
        return ''.join([self.id_to_token.get(id, '') for id in ids if id != self.blank_id])

class SubwordTokenizer(TokenizerBase):
    """Subword-level tokenizer using SentencePiece BPE."""

    def __init__(self, name: str = "subword", vocab_size: int = 5000, model_path: Optional[str] = None):
        super().__init__(name, vocab_size)
        self.sp_model = None
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            try:
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.Load(model_path)
                self.vocab_size = self.sp_model.GetPieceSize()

                # Create mapping
                self.id_to_token = {i: self.sp_model.IdToPiece(i) for i in range(self.vocab_size)}
                self.token_to_id = {v: k for k, v in self.id_to_token.items()}

                # Set blank ID
                self.blank_id = self.vocab_size
                print(f"Successfully loaded SentencePiece model from {model_path} with {self.vocab_size} tokens")
            except Exception as e:
                print(f"Error loading SentencePiece model from {model_path}: {e}")
                print("Creating a simple character-based fallback model")
                self._create_fallback_model()
        else:
            print("No SentencePiece model provided. Creating a simple character-based fallback model")
            self._create_fallback_model()

    def _create_fallback_model(self):
        """Create a simple character-based fallback model."""
        # Vietnamese characters (lowercase)
        base_chars = "abcdefghijklmnopqrstuvwxyz"
        # Vietnamese diacritics
        vn_chars = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
        # Digits and special characters
        digits = "0123456789"
        special = " ,.!?-:;\"'()[]{}%$#@&*+=/<>|~^_"

        # Combine all characters
        all_chars = base_chars + vn_chars + digits + special

        # Create vocabulary
        self.id_to_token = {i: c for i, c in enumerate(all_chars)}
        self.token_to_id = {c: i for i, c in enumerate(all_chars)}
        self.vocab_size = len(self.id_to_token)

        # Add blank token at the end
        self.blank_id = self.vocab_size
        self.id_to_token[self.blank_id] = BLANK_TOKEN
        self.token_to_id[BLANK_TOKEN] = self.blank_id

        # Create a dummy sp_model that just uses character tokenization
        self.sp_model = None

    def train(self, text_file: str, model_prefix: str) -> None:
        """Train a SentencePiece model."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

            # Check if text file exists
            if not os.path.exists(text_file):
                print(f"Text file {text_file} not found. Creating a sample file...")
                with open(text_file, 'w', encoding='utf-8') as f:
                    # Sample Vietnamese sentences
                    sentences = [
                        "Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói.",
                        "Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây.",
                        "Mô hình CTC đã cải thiện đáng kể độ chính xác của hệ thống nhận dạng tiếng nói.",
                        "Việt Nam có một nền văn hóa phong phú và đa dạng với nhiều dân tộc khác nhau.",
                        "Chúng tôi đang nghiên cứu các phương pháp cải thiện độ chính xác của hệ thống ASR."
                    ]
                    f.write('\n'.join(sentences))

            # Train the model
            spm.SentencePieceTrainer.Train(
                f'--input={text_file} '
                f'--model_prefix={model_prefix} '
                f'--vocab_size={self.vocab_size} '
                f'--character_coverage=1.0 '
                f'--model_type=bpe '
                f'--pad_id=-1 '
                f'--unk_id=0 '
                f'--bos_id=1 '
                f'--eos_id=2 '
                f'--normalization_rule_name=identity'
            )

            # Load the trained model
            model_path = f"{model_prefix}.model"
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(model_path)

            # Create mapping
            self.vocab_size = self.sp_model.GetPieceSize()
            self.id_to_token = {i: self.sp_model.IdToPiece(i) for i in range(self.vocab_size)}
            self.token_to_id = {v: k for k, v in self.id_to_token.items()}

            # Set blank ID
            self.blank_id = self.vocab_size

            print(f"Successfully trained and loaded SentencePiece model with {self.vocab_size} tokens")
        except Exception as e:
            print(f"Error training SentencePiece model: {e}")
            print("Creating a simple character-based fallback model")
            self._create_fallback_model()

    def encode(self, text: str) -> List[int]:
        """Convert text to subword IDs."""
        if self.sp_model is not None:
            try:
                return self.sp_model.EncodeAsIds(text.lower())
            except Exception as e:
                print(f"Error encoding text with SentencePiece: {e}")
                print("Falling back to character-level encoding")
                # Fallback to character-level encoding
                return [self.token_to_id.get(c, self.token_to_id.get(UNK_TOKEN, 0)) for c in text.lower()]
        else:
            # Character-level encoding
            return [self.token_to_id.get(c, self.token_to_id.get(UNK_TOKEN, 0)) for c in text.lower()]

    def decode(self, ids: List[int]) -> str:
        """Convert subword IDs to text."""
        # Filter out blank tokens
        filtered_ids = [id for id in ids if id != self.blank_id]

        if self.sp_model is not None:
            try:
                return self.sp_model.DecodeIds(filtered_ids)
            except Exception as e:
                print(f"Error decoding IDs with SentencePiece: {e}")
                print("Falling back to character-level decoding")
                # Fallback to character-level decoding
                return ''.join([self.id_to_token.get(id, '') for id in filtered_ids])
        else:
            # Character-level decoding
            return ''.join([self.id_to_token.get(id, '') for id in filtered_ids])

class SyllableTokenizer(TokenizerBase):
    """Syllable-level tokenizer for Vietnamese."""

    def __init__(self, name: str = "syllable", vocab_size: int = 0, vocab_file: Optional[str] = None):
        super().__init__(name, vocab_size)

        if vocab_file and os.path.exists(vocab_file):
            try:
                # Load syllable vocabulary from file
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    syllables = [line.strip() for line in f]

                # Create vocabulary
                self.id_to_token = {i: s for i, s in enumerate(syllables)}
                self.token_to_id = {s: i for i, s in enumerate(syllables)}
                self.vocab_size = len(self.id_to_token)

                # Add special tokens
                self.unk_id = self.vocab_size
                self.id_to_token[self.unk_id] = UNK_TOKEN
                self.token_to_id[UNK_TOKEN] = self.unk_id

                # Add blank token at the end
                self.blank_id = self.vocab_size + 1
                self.id_to_token[self.blank_id] = BLANK_TOKEN
                self.token_to_id[BLANK_TOKEN] = self.blank_id

                print(f"Loaded syllable vocabulary with {self.vocab_size} syllables")
            except Exception as e:
                print(f"Error loading syllable vocabulary: {e}")
                self._create_fallback_vocabulary()
        else:
            print(f"Syllable vocabulary file not found: {vocab_file}")
            self._create_fallback_vocabulary()

    def _create_fallback_vocabulary(self):
        """Create a simple fallback vocabulary."""
        # Common Vietnamese syllables
        syllables = [
            "xin", "chào", "tôi", "là", "người", "việt", "nam", "học", "tiếng", "anh",
            "cảm", "ơn", "bạn", "rất", "vui", "được", "gặp", "nhà", "trường", "sinh",
            "viên", "giáo", "dục", "công", "nghệ", "thông", "tin", "khoa", "học", "máy",
            "tính", "phần", "mềm", "hệ", "thống", "mạng", "internet", "điện", "thoại", "di",
            "động", "thời", "gian", "ngày", "tháng", "năm", "giờ", "phút", "giây", "sáng",
            "trưa", "chiều", "tối", "đêm", "hôm", "nay", "mai", "qua", "kia", "mốt"
        ]

        # Create vocabulary
        self.id_to_token = {i: s for i, s in enumerate(syllables)}
        self.token_to_id = {s: i for i, s in enumerate(syllables)}
        self.vocab_size = len(self.id_to_token)

        # Add special tokens
        self.unk_id = self.vocab_size
        self.id_to_token[self.unk_id] = UNK_TOKEN
        self.token_to_id[UNK_TOKEN] = self.unk_id

        # Add blank token at the end
        self.blank_id = self.vocab_size + 1
        self.id_to_token[self.blank_id] = BLANK_TOKEN
        self.token_to_id[BLANK_TOKEN] = self.blank_id

        print(f"Created fallback syllable vocabulary with {self.vocab_size} syllables")

    def encode(self, text: str) -> List[int]:
        """Convert text to syllable IDs."""
        try:
            # Split text into syllables (words separated by spaces in Vietnamese)
            syllables = text.lower().split()
            return [self.token_to_id.get(s, self.unk_id) for s in syllables]
        except Exception as e:
            print(f"Error in syllable encoding: {e}")
            # Return a sequence of UNK tokens as a last resort
            return [self.unk_id] * (len(text.split()) or 1)

    def decode(self, ids: List[int]) -> str:
        """Convert syllable IDs to text."""
        try:
            return ' '.join([self.id_to_token.get(id, '') for id in ids if id != self.blank_id])
        except Exception as e:
            print(f"Error in syllable decoding: {e}")
            return UNK_TOKEN  # Return a single UNK token as a last resort

class WordTokenizer(TokenizerBase):
    """Word-level tokenizer for Vietnamese with OOV fallback."""

    def __init__(self, name: str = "word", vocab_size: int = 20000,
                 vocab_file: Optional[str] = None,
                 subword_tokenizer: Optional[SubwordTokenizer] = None):
        super().__init__(name, vocab_size)
        self.subword_tokenizer = subword_tokenizer

        if vocab_file and os.path.exists(vocab_file):
            try:
                # Load word vocabulary from file
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    words = [line.strip() for line in f]

                # Create vocabulary
                self.id_to_token = {i: w for i, w in enumerate(words)}
                self.token_to_id = {w: i for i, w in enumerate(words)}
                self.vocab_size = len(self.id_to_token)

                # Add special tokens
                self.unk_id = self.vocab_size
                self.id_to_token[self.unk_id] = UNK_TOKEN
                self.token_to_id[UNK_TOKEN] = self.unk_id

                # Add blank token at the end
                self.blank_id = self.vocab_size + 1
                self.id_to_token[self.blank_id] = BLANK_TOKEN
                self.token_to_id[BLANK_TOKEN] = self.blank_id

                print(f"Loaded word vocabulary with {self.vocab_size} words")
            except Exception as e:
                print(f"Error loading word vocabulary: {e}")
                self._create_fallback_vocabulary()
        else:
            print(f"Word vocabulary file not found: {vocab_file}")
            self._create_fallback_vocabulary()

    def _create_fallback_vocabulary(self):
        """Create a simple fallback vocabulary."""
        # Common Vietnamese words
        words = [
            "xin", "chào", "tôi", "là", "người", "việt", "nam", "học", "tiếng", "anh",
            "cảm", "ơn", "bạn", "rất", "vui", "được", "gặp", "nhà", "trường", "sinh",
            "viên", "giáo", "dục", "công", "nghệ", "thông", "tin", "khoa", "học", "máy",
            "tính", "phần", "mềm", "hệ", "thống", "mạng", "internet", "điện", "thoại", "di",
            "động", "thời", "gian", "ngày", "tháng", "năm", "giờ", "phút", "giây", "sáng",
            "xin chào", "cảm ơn", "việt nam", "học sinh", "sinh viên", "giáo dục",
            "công nghệ", "thông tin", "khoa học", "máy tính"
        ]

        # Create vocabulary
        self.id_to_token = {i: w for i, w in enumerate(words)}
        self.token_to_id = {w: i for i, w in enumerate(words)}
        self.vocab_size = len(self.id_to_token)

        # Add special tokens
        self.unk_id = self.vocab_size
        self.id_to_token[self.unk_id] = UNK_TOKEN
        self.token_to_id[UNK_TOKEN] = self.unk_id

        # Add blank token at the end
        self.blank_id = self.vocab_size + 1
        self.id_to_token[self.blank_id] = BLANK_TOKEN
        self.token_to_id[BLANK_TOKEN] = self.blank_id

        print(f"Created fallback word vocabulary with {self.vocab_size} words")

    def encode(self, text: str) -> List[int]:
        """Convert text to word IDs with OOV handling."""
        try:
            # Split text into words (multi-syllable words in Vietnamese)
            # This is a simplification; proper Vietnamese word segmentation is more complex
            words = text.lower().split()

            ids = []
            for word in words:
                if word in self.token_to_id:
                    ids.append(self.token_to_id[word])
                else:
                    # For OOV words, just use the UNK token
                    # This is simpler and safer than using subword tokenization with offsets
                    ids.append(self.unk_id)

            return ids
        except Exception as e:
            print(f"Error in word encoding: {e}")
            # Return a sequence of UNK tokens as a last resort
            return [self.unk_id] * (len(text.split()) or 1)

    def decode(self, ids: List[int]) -> str:
        """Convert word IDs to text with OOV handling."""
        try:
            words = []
            for id in ids:
                if id == self.blank_id:
                    continue
                elif id < self.vocab_size:
                    # Regular word
                    words.append(self.id_to_token.get(id, ''))
                elif id == self.unk_id:
                    # Unknown word
                    words.append(UNK_TOKEN)
                else:
                    # Unrecognized ID
                    words.append(UNK_TOKEN)

            return ' '.join(words)
        except Exception as e:
            print(f"Error in word decoding: {e}")
            return UNK_TOKEN  # Return a single UNK token as a last resort

class CTCDecoder:
    """CTC decoder implementation with greedy and beam search algorithms."""

    def __init__(self, tokenizer: TokenizerBase):
        self.tokenizer = tokenizer
        self.blank_id = tokenizer.get_blank_id()

        # Make sure blank_id is valid
        vocab_size = tokenizer.get_vocab_size()
        if self.blank_id >= vocab_size:
            print(f"Warning: Blank ID {self.blank_id} is out of bounds for vocabulary size {vocab_size}. Using last token as blank.")
            self.blank_id = vocab_size - 1

    def greedy_decode(self, logits: torch.Tensor) -> Tuple[List[int], float]:
        """
        Greedy CTC decoding.

        Args:
            logits: Tensor of shape [T, V] where T is the sequence length and V is the vocabulary size

        Returns:
            Tuple of (decoded token IDs, decoding time in seconds)
        """
        start_time = time.time()

        # Get the most likely token at each timestep
        predictions = torch.argmax(logits, dim=1).tolist()

        # Apply CTC decoding rules:
        # 1. Remove repeated tokens
        # 2. Remove blank tokens
        decoded = []
        prev_token = -1
        for token in predictions:
            if token != prev_token and token != self.blank_id:
                decoded.append(token)
            prev_token = token

        decoding_time = time.time() - start_time
        return decoded, decoding_time

    def beam_search_decode(self, logits: torch.Tensor, lm: Optional[Any] = None,
                          beam_size: int = 10, alpha: float = 0.5, beta: float = 1.0) -> Tuple[List[int], float]:
        """
        Beam search CTC decoding with optional language model integration.

        Args:
            logits: Tensor of shape [T, V] where T is the sequence length and V is the vocabulary size
            lm: Language model (KenLM model)
            beam_size: Beam size
            alpha: Language model weight
            beta: Length penalty

        Returns:
            Tuple of (decoded token IDs, decoding time in seconds)
        """
        if not KENLM_AVAILABLE and lm is not None:
            print("Warning: KenLM not available. Falling back to greedy decoding.")
            return self.greedy_decode(logits)

        start_time = time.time()

        # Convert logits to log probabilities
        log_probs = torch.log_softmax(logits, dim=1)
        T, V = log_probs.shape

        # Initialize beam with empty sequence
        beam = [([], 0.0)]  # (prefix, score)

        # Process each timestep
        for t in range(T):
            new_beam = {}

            for prefix, score in beam:
                # Option 1: Add blank (no new token)
                blank_score = score + log_probs[t, self.blank_id].item()
                prefix_str = tuple(prefix)
                new_beam[prefix_str] = max(new_beam.get(prefix_str, float('-inf')), blank_score)

                # Option 2: Add non-blank tokens
                for v in range(V):
                    if v == self.blank_id:
                        continue

                    # Apply CTC rules
                    new_prefix = list(prefix)
                    if len(prefix) == 0 or v != prefix[-1]:
                        # New token is different from the last one
                        new_prefix = new_prefix + [v]

                    # Calculate new score
                    new_score = score + log_probs[t, v].item()

                    # Apply language model if available
                    if lm is not None:
                        prefix_text = self.tokenizer.decode(new_prefix)
                        lm_score = lm.score(prefix_text, bos=True, eos=False)
                        new_score += alpha * lm_score

                    # Apply length penalty
                    new_score -= beta * len(new_prefix)

                    # Update beam
                    new_prefix_str = tuple(new_prefix)
                    new_beam[new_prefix_str] = max(new_beam.get(new_prefix_str, float('-inf')), new_score)

            # Keep only top-k beams
            beam = sorted([(list(prefix), score) for prefix, score in new_beam.items()],
                         key=lambda x: x[1], reverse=True)[:beam_size]

        # Return the best path
        best_prefix, _ = beam[0]

        decoding_time = time.time() - start_time
        return best_prefix, decoding_time

class ASREvaluator:
    """Evaluator for ASR systems with different tokenization levels."""

    def __init__(self, audio_dir: str, transcript_dir: str, output_dir: str):
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.output_dir = output_dir
        self.tokenizers = {}
        self.lm_models = {}
        self.results = defaultdict(list)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def add_tokenizer(self, tokenizer: TokenizerBase, lm_path: Optional[str] = None):
        """Add a tokenizer for evaluation."""
        self.tokenizers[tokenizer.name] = tokenizer

        # Load language model if available
        if lm_path and os.path.exists(lm_path) and KENLM_AVAILABLE:
            self.lm_models[tokenizer.name] = kenlm.Model(lm_path)
        else:
            self.lm_models[tokenizer.name] = None

    def prepare_test_data(self, num_samples: int = 10, max_length: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Prepare test data for evaluation.

        Args:
            num_samples: Number of samples to use
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping file IDs to data dictionaries
        """
        test_data = {}

        # Check if we have any tokenizers
        if not self.tokenizers:
            print("Error: No tokenizers available. Please add at least one tokenizer before preparing test data.")
            return test_data

        # Get audio files
        try:
            audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
            if num_samples > 0:
                audio_files = audio_files[:num_samples]

            if not audio_files:
                print(f"Warning: No audio files found in {self.audio_dir}. Creating sample data...")
                self._create_sample_data()
                audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith('.wav')]
                if num_samples > 0:
                    audio_files = audio_files[:num_samples]

            for audio_file in tqdm(audio_files, desc="Preparing test data"):
                try:
                    file_id = os.path.splitext(audio_file)[0]
                    audio_path = os.path.join(self.audio_dir, audio_file)
                    transcript_path = os.path.join(self.transcript_dir, f"{file_id}.txt")

                    if not os.path.exists(transcript_path):
                        print(f"Warning: No transcript found for {audio_file}. Skipping.")
                        continue

                    # Load audio
                    try:
                        waveform, sample_rate = torchaudio.load(audio_path)
                    except Exception as e:
                        print(f"Error loading audio file {audio_path}: {e}")
                        print("Creating a dummy waveform")
                        waveform = torch.zeros(1, 16000)  # 1 second of silence at 16kHz
                        sample_rate = 16000

                    # Load transcript
                    try:
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript = f.read().strip()
                    except Exception as e:
                        print(f"Error loading transcript file {transcript_path}: {e}")
                        print("Using a dummy transcript")
                        transcript = "Dummy transcript for testing"

                    # Generate mock logits for testing
                    # In a real scenario, these would come from an acoustic model
                    mock_logits = self._generate_mock_logits(transcript, max_length)

                    test_data[file_id] = {
                        'audio_path': audio_path,
                        'transcript': transcript,
                        'waveform': waveform,
                        'sample_rate': sample_rate,
                        'logits': mock_logits
                    }
                except Exception as e:
                    print(f"Error processing file {audio_file}: {e}")
                    continue
        except Exception as e:
            print(f"Error preparing test data: {e}")
            print("Creating sample data for testing...")
            self._create_sample_data()
            return self.prepare_test_data(num_samples, max_length)

        return test_data

    def _create_sample_data(self, num_samples: int = 5):
        """Create sample audio and transcript files for testing."""
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)

        # Sample Vietnamese sentences
        sentences = [
            "Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói.",
            "Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây.",
            "Mô hình CTC đã cải thiện đáng kể độ chính xác của hệ thống nhận dạng tiếng nói.",
            "Việt Nam có một nền văn hóa phong phú và đa dạng với nhiều dân tộc khác nhau.",
            "Chúng tôi đang nghiên cứu các phương pháp cải thiện độ chính xác của hệ thống ASR."
        ]

        # Create sample audio files (1-second silence) and transcripts
        for i in range(min(num_samples, len(sentences))):
            # Create a simple audio file (1 second of silence at 16kHz)
            sample_rate = 16000
            waveform = torch.zeros(1, sample_rate)  # 1 second of silence

            # Save audio file
            audio_path = os.path.join(self.audio_dir, f"sample_{i+1:02d}.wav")
            try:
                torchaudio.save(audio_path, waveform, sample_rate)
            except Exception as e:
                print(f"Error saving audio file {audio_path}: {e}")
                # Create an empty file as a fallback
                with open(audio_path, 'wb') as f:
                    f.write(b'')

            # Save transcript
            transcript_path = os.path.join(self.transcript_dir, f"sample_{i+1:02d}.txt")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(sentences[i])

        print(f"Created {min(num_samples, len(sentences))} sample audio files and transcripts.")

    def _generate_mock_logits(self, transcript: str, max_length: int) -> Dict[str, torch.Tensor]:
        """
        Generate mock logits for testing.

        Args:
            transcript: Reference transcript
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping tokenizer names to logits tensors
        """
        mock_logits = {}

        for name, tokenizer in self.tokenizers.items():
            try:
                # Encode transcript
                token_ids = tokenizer.encode(transcript)

                # Filter out token IDs that are out of bounds
                # This is especially important for word tokenizer with OOV handling
                vocab_size = tokenizer.get_vocab_size()
                filtered_token_ids = []

                for token_id in token_ids:
                    if 0 <= token_id < vocab_size:
                        filtered_token_ids.append(token_id)
                    else:
                        # For OOV tokens in word tokenizer, use the UNK token instead
                        print(f"Warning: Token ID {token_id} is out of bounds for vocabulary size {vocab_size}. Using UNK token instead.")
                        if hasattr(tokenizer, 'unk_id'):
                            filtered_token_ids.append(tokenizer.unk_id)
                        else:
                            # If no UNK token is defined, use a random valid token
                            filtered_token_ids.append(np.random.randint(0, vocab_size))

                # If no valid tokens remain, use a simple sequence of UNK tokens
                if not filtered_token_ids:
                    print(f"Warning: No valid tokens found for {name} tokenizer. Using UNK tokens.")
                    if hasattr(tokenizer, 'unk_id') and 0 <= tokenizer.unk_id < vocab_size:
                        filtered_token_ids = [tokenizer.unk_id] * 5  # Use 5 UNK tokens as a fallback
                    else:
                        filtered_token_ids = [0] * 5  # Use token ID 0 as a fallback

                # Create sequence length (3x the number of tokens, to simulate CTC behavior)
                seq_length = min(len(filtered_token_ids) * 3, max_length)

                # Initialize random logits
                logits = torch.randn(seq_length, vocab_size)

                # Bias logits towards the correct tokens
                # This simulates a trained model's output
                blank_id = tokenizer.get_blank_id()

                # Make sure blank_id is within bounds
                if blank_id >= vocab_size:
                    print(f"Warning: Blank ID {blank_id} is out of bounds for vocabulary size {vocab_size}. Using last token as blank.")
                    blank_id = vocab_size - 1

                # Distribute token IDs across the sequence with blanks in between
                token_positions = np.linspace(0, seq_length-1, len(filtered_token_ids), dtype=int)

                for i, pos in enumerate(token_positions):
                    token_id = filtered_token_ids[i]
                    # Make the correct token more likely
                    logits[pos, token_id] += 5.0

                    # Add some blanks between tokens
                    if i < len(token_positions) - 1:
                        next_pos = token_positions[i+1]
                        for blank_pos in range(pos+1, next_pos):
                            logits[blank_pos, blank_id] += 3.0

                mock_logits[name] = logits
            except Exception as e:
                print(f"Error generating mock logits for {name} tokenizer: {e}")
                # Create a simple fallback logits tensor
                vocab_size = tokenizer.get_vocab_size()
                seq_length = min(len(transcript) * 3, max_length)
                logits = torch.randn(seq_length, vocab_size)
                mock_logits[name] = logits
                print(f"Created fallback logits tensor with shape {logits.shape}")

        return mock_logits

    def evaluate(self, test_data: Dict[str, Dict[str, Any]],
                beam_size: int = 10, alpha: float = 0.5, beta: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate ASR performance with different tokenization levels.

        Args:
            test_data: Test data dictionary
            beam_size: Beam size for beam search decoding
            alpha: Language model weight
            beta: Length penalty

        Returns:
            Dictionary of evaluation results
        """
        results = {}

        for tokenizer_name, tokenizer in self.tokenizers.items():
            print(f"\nEvaluating {tokenizer_name} tokenizer...")
            decoder = CTCDecoder(tokenizer)
            lm = self.lm_models.get(tokenizer_name)

            # Initialize metrics
            metrics = {
                'cer': [],
                'wer': [],
                'ser': [],
                'greedy_time': [],
                'beam_time': [],
                'samples': []
            }

            for file_id, data in tqdm(test_data.items(), desc=f"Decoding with {tokenizer_name}"):
                transcript = data['transcript']
                logits = data['logits'][tokenizer_name]

                # Greedy decoding
                greedy_ids, greedy_time = decoder.greedy_decode(logits)
                greedy_text = tokenizer.decode(greedy_ids)

                # Beam search decoding
                beam_ids, beam_time = decoder.beam_search_decode(
                    logits, lm, beam_size=beam_size, alpha=alpha, beta=beta
                )
                beam_text = tokenizer.decode(beam_ids)

                # Calculate metrics
                try:
                    # Ensure transcript and hypothesis are not empty
                    if not transcript or not beam_text:
                        raise ValueError("Reference or hypothesis is empty")

                    # Character Error Rate
                    sample_cer = cer(transcript, beam_text)

                    # For word-level metrics, we need to use the original strings
                    # jiwer library handles tokenization internally
                    sample_wer = wer(transcript, beam_text)

                    # For syllable-level metrics (specific to Vietnamese)
                    # In Vietnamese, syllables are separated by spaces, so WER = SER
                    sample_ser = sample_wer  # Same as WER for Vietnamese
                except Exception as e:
                    print(f"Error calculating metrics for {file_id}: {e}")
                    print(f"Reference: '{transcript}'")
                    print(f"Hypothesis: '{beam_text}'")
                    # Use default values in case of error
                    sample_cer = 1.0  # 100% error
                    sample_wer = 1.0  # 100% error
                    sample_ser = 1.0  # 100% error

                # Store results
                metrics['cer'].append(sample_cer)
                metrics['wer'].append(sample_wer)
                metrics['ser'].append(sample_ser)
                metrics['greedy_time'].append(greedy_time)
                metrics['beam_time'].append(beam_time)

                # Store sample results for detailed analysis
                metrics['samples'].append({
                    'file_id': file_id,
                    'reference': transcript,
                    'greedy_hypothesis': greedy_text,
                    'beam_hypothesis': beam_text,
                    'cer': sample_cer,
                    'wer': sample_wer,
                    'ser': sample_ser,
                    'greedy_time': greedy_time,
                    'beam_time': beam_time
                })

            # Calculate average metrics
            results[tokenizer_name] = {
                'avg_cer': np.mean(metrics['cer']),
                'avg_wer': np.mean(metrics['wer']),
                'avg_ser': np.mean(metrics['ser']),
                'avg_greedy_time': np.mean(metrics['greedy_time']),
                'avg_beam_time': np.mean(metrics['beam_time']),
                'samples': metrics['samples']
            }

            # Print summary
            print(f"  Average CER: {results[tokenizer_name]['avg_cer']:.4f}")
            print(f"  Average WER: {results[tokenizer_name]['avg_wer']:.4f}")
            print(f"  Average SER: {results[tokenizer_name]['avg_ser']:.4f}")
            print(f"  Average greedy decoding time: {results[tokenizer_name]['avg_greedy_time']*1000:.2f} ms")
            print(f"  Average beam search decoding time: {results[tokenizer_name]['avg_beam_time']*1000:.2f} ms")

        return results

    def generate_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a detailed report of the evaluation results.

        Args:
            results: Evaluation results

        Returns:
            Path to the generated report
        """
        # Create report directory
        report_dir = os.path.join(self.output_dir, 'report')
        os.makedirs(report_dir, exist_ok=True)

        # Create summary table
        summary_data = []
        for tokenizer_name, result in results.items():
            summary_data.append({
                'Tokenizer': tokenizer_name,
                'CER': f"{result['avg_cer']:.4f}",
                'WER': f"{result['avg_wer']:.4f}",
                'SER': f"{result['avg_ser']:.4f}",
                'Greedy Time (ms)': f"{result['avg_greedy_time']*1000:.2f}",
                'Beam Time (ms)': f"{result['avg_beam_time']*1000:.2f}"
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary table
        summary_path = os.path.join(report_dir, 'summary.csv')
        summary_df.to_csv(summary_path, index=False)

        # Create comparison plots
        self._create_error_rate_plot(results, os.path.join(report_dir, 'error_rates.png'))
        self._create_latency_plot(results, os.path.join(report_dir, 'latency.png'))
        self._create_tradeoff_plot(results, os.path.join(report_dir, 'tradeoff.png'))

        # Generate HTML report
        html_path = os.path.join(report_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_html_report(results, summary_df))

        # Generate Markdown report
        md_path = os.path.join(report_dir, 'report.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(results, summary_df))

        print(f"\nReport generated at {report_dir}")
        return report_dir

    def _create_error_rate_plot(self, results: Dict[str, Dict[str, Any]], output_path: str) -> None:
        """Create a plot comparing error rates across tokenization levels."""
        plt.figure(figsize=(10, 6))

        tokenizers = list(results.keys())
        cer_values = [results[t]['avg_cer'] for t in tokenizers]
        wer_values = [results[t]['avg_wer'] for t in tokenizers]
        ser_values = [results[t]['avg_ser'] for t in tokenizers]

        x = np.arange(len(tokenizers))
        width = 0.25

        plt.bar(x - width, cer_values, width, label='CER')
        plt.bar(x, wer_values, width, label='WER')
        plt.bar(x + width, ser_values, width, label='SER')

        plt.xlabel('Tokenization Level')
        plt.ylabel('Error Rate')
        plt.title('Error Rates by Tokenization Level')
        plt.xticks(x, tokenizers)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _create_latency_plot(self, results: Dict[str, Dict[str, Any]], output_path: str) -> None:
        """Create a plot comparing decoding latency across tokenization levels."""
        plt.figure(figsize=(10, 6))

        tokenizers = list(results.keys())
        greedy_times = [results[t]['avg_greedy_time'] * 1000 for t in tokenizers]  # Convert to ms
        beam_times = [results[t]['avg_beam_time'] * 1000 for t in tokenizers]  # Convert to ms

        x = np.arange(len(tokenizers))
        width = 0.35

        plt.bar(x - width/2, greedy_times, width, label='Greedy Decoding')
        plt.bar(x + width/2, beam_times, width, label='Beam Search')

        plt.xlabel('Tokenization Level')
        plt.ylabel('Latency (ms)')
        plt.title('Decoding Latency by Tokenization Level')
        plt.xticks(x, tokenizers)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _create_tradeoff_plot(self, results: Dict[str, Dict[str, Any]], output_path: str) -> None:
        """Create a plot showing the trade-off between accuracy and speed."""
        plt.figure(figsize=(10, 6))

        tokenizers = list(results.keys())
        cer_values = [results[t]['avg_cer'] for t in tokenizers]
        latency_values = [results[t]['avg_beam_time'] * 1000 for t in tokenizers]  # Convert to ms

        # Create scatter plot
        plt.scatter(latency_values, cer_values, s=100)

        # Add labels for each point
        for i, txt in enumerate(tokenizers):
            plt.annotate(txt, (latency_values[i], cer_values[i]),
                        xytext=(10, 5), textcoords='offset points')

        plt.xlabel('Latency (ms)')
        plt.ylabel('Character Error Rate (CER)')
        plt.title('Accuracy vs. Speed Trade-off')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _generate_html_report(self, results: Dict[str, Dict[str, Any]], summary_df: pd.DataFrame) -> str:
        """Generate an HTML report of the evaluation results."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vietnamese ASR Tokenization Evaluation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .plot { margin: 20px 0; max-width: 100%; }
                .recommendations { background-color: #f8f8f8; padding: 15px; border-left: 4px solid #4CAF50; }
            </style>
        </head>
        <body>
            <h1>Vietnamese ASR Tokenization Evaluation Report</h1>

            <h2>Summary</h2>
            <table>
                <tr>
        """

        # Add table headers
        for col in summary_df.columns:
            html += f"<th>{col}</th>"
        html += "</tr>"

        # Add table rows
        for _, row in summary_df.iterrows():
            html += "<tr>"
            for col in summary_df.columns:
                html += f"<td>{row[col]}</td>"
            html += "</tr>"

        html += """
            </table>

            <h2>Comparison Plots</h2>
            <div class="plot">
                <h3>Error Rates by Tokenization Level</h3>
                <img src="error_rates.png" alt="Error Rates" width="800">
            </div>

            <div class="plot">
                <h3>Decoding Latency by Tokenization Level</h3>
                <img src="latency.png" alt="Latency" width="800">
            </div>

            <div class="plot">
                <h3>Accuracy vs. Speed Trade-off</h3>
                <img src="tradeoff.png" alt="Trade-off" width="800">
            </div>

            <h2>Analysis and Recommendations</h2>
            <div class="recommendations">
                <h3>Tokenization Level Comparison</h3>
                <ul>
                    <li><strong>Character-level:</strong>
                        <ul>
                            <li>Pros: Smallest vocabulary, no OOV issues, handles all Vietnamese characters</li>
                            <li>Cons: Longer sequences, slower decoding, requires stronger language model</li>
                            <li>Best for: Maximum coverage of Vietnamese text, handling rare words</li>
                        </ul>
                    </li>
                    <li><strong>Subword-level (BPE):</strong>
                        <ul>
                            <li>Pros: Good balance between vocabulary size and sequence length, handles word variations</li>
                            <li>Cons: May produce suboptimal segmentation for Vietnamese</li>
                            <li>Best for: General-purpose ASR with good balance of accuracy and speed</li>
                        </ul>
                    </li>
                    <li><strong>Syllable-level:</strong>
                        <ul>
                            <li>Pros: Natural unit for Vietnamese, shorter sequences than characters</li>
                            <li>Cons: Larger vocabulary than characters, potential OOV issues</li>
                            <li>Best for: Vietnamese-specific ASR where syllables are well-defined</li>
                        </ul>
                    </li>
                    <li><strong>Word-level:</strong>
                        <ul>
                            <li>Pros: Shortest sequences, fastest decoding, direct mapping to meaning</li>
                            <li>Cons: Largest vocabulary, significant OOV issues, requires word segmentation</li>
                            <li>Best for: Domain-specific applications with limited vocabulary, real-time requirements</li>
                        </ul>
                    </li>
                </ul>

                <h3>Recommendations</h3>
                <ul>
                    <li><strong>For real-time applications:</strong> Consider syllable or word-level tokenization with greedy decoding</li>
                    <li><strong>For maximum accuracy:</strong> Use character or subword-level with beam search and a strong LM</li>
                    <li><strong>For Vietnamese-specific applications:</strong> Syllable-level tokenization offers a good balance</li>
                    <li><strong>For general-purpose ASR:</strong> Subword-level (BPE) with 5,000-10,000 tokens provides flexibility</li>
                </ul>
            </div>
        </body>
        </html>
        """

        return html

    def _generate_markdown_report(self, results: Dict[str, Dict[str, Any]], summary_df: pd.DataFrame) -> str:
        """Generate a Markdown report of the evaluation results."""
        md = """# Vietnamese ASR Tokenization Evaluation Report

## Summary

"""
        # Add table
        md += "| " + " | ".join(summary_df.columns) + " |\n"
        md += "| " + " | ".join(["---" for _ in summary_df.columns]) + " |\n"

        for _, row in summary_df.iterrows():
            md += "| " + " | ".join([str(row[col]) for col in summary_df.columns]) + " |\n"

        md += """
## Comparison Plots

### Error Rates by Tokenization Level
![Error Rates](error_rates.png)

### Decoding Latency by Tokenization Level
![Latency](latency.png)

### Accuracy vs. Speed Trade-off
![Trade-off](tradeoff.png)

## Analysis and Recommendations

### Tokenization Level Comparison

- **Character-level:**
  - Pros: Smallest vocabulary, no OOV issues, handles all Vietnamese characters
  - Cons: Longer sequences, slower decoding, requires stronger language model
  - Best for: Maximum coverage of Vietnamese text, handling rare words

- **Subword-level (BPE):**
  - Pros: Good balance between vocabulary size and sequence length, handles word variations
  - Cons: May produce suboptimal segmentation for Vietnamese
  - Best for: General-purpose ASR with good balance of accuracy and speed

- **Syllable-level:**
  - Pros: Natural unit for Vietnamese, shorter sequences than characters
  - Cons: Larger vocabulary than characters, potential OOV issues
  - Best for: Vietnamese-specific ASR where syllables are well-defined

- **Word-level:**
  - Pros: Shortest sequences, fastest decoding, direct mapping to meaning
  - Cons: Largest vocabulary, significant OOV issues, requires word segmentation
  - Best for: Domain-specific applications with limited vocabulary, real-time requirements

### Recommendations

- **For real-time applications:** Consider syllable or word-level tokenization with greedy decoding
- **For maximum accuracy:** Use character or subword-level with beam search and a strong LM
- **For Vietnamese-specific applications:** Syllable-level tokenization offers a good balance
- **For general-purpose ASR:** Subword-level (BPE) with 5,000-10,000 tokens provides flexibility
"""

        return md

def main():
    """Main function to run the CTC decoder evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate CTC decoders for Vietnamese ASR")
    parser.add_argument("--audio_dir", type=str, default="data/audio",
                       help="Directory containing audio files")
    parser.add_argument("--transcript_dir", type=str, default="data/transcripts",
                       help="Directory containing transcript files")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to evaluate")
    parser.add_argument("--beam_size", type=int, default=10,
                       help="Beam size for beam search decoding")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Language model weight")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Length penalty")

    # Tokenizer options
    parser.add_argument("--tokenizers", type=str, default="all",
                       help="Comma-separated list of tokenizers to use: 'character', 'subword', 'syllable', 'word', or 'all' (default)")
    parser.add_argument("--subword_vocab_size", type=int, default=5000,
                       help="Vocabulary size for subword tokenizer (default: 5000)")
    parser.add_argument("--subword_model_path", type=str, default="data/tokenizers/vietnamese_bpe_5000.model",
                       help="Path to SentencePiece model for subword tokenization")
    parser.add_argument("--syllable_vocab_path", type=str, default="data/tokenizers/vietnamese_syllables.txt",
                       help="Path to syllable vocabulary file")
    parser.add_argument("--word_vocab_path", type=str, default="data/tokenizers/vietnamese_words.txt",
                       help="Path to word vocabulary file")

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.transcript_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data/tokenizers", exist_ok=True)
    os.makedirs("data/lm", exist_ok=True)

    # Initialize evaluator
    evaluator = ASREvaluator(args.audio_dir, args.transcript_dir, args.output_dir)

    # Parse tokenizer options
    tokenizer_options = args.tokenizers.lower().split(',')
    if 'all' in tokenizer_options:
        tokenizer_options = ['character', 'subword', 'syllable', 'word']

    print(f"Initializing tokenizers: {', '.join(tokenizer_options)}")

    # Initialize subword tokenizer first (needed for word tokenizer's OOV handling)
    subword_tokenizer = None
    if 'subword' in tokenizer_options or 'word' in tokenizer_options:
        # Subword tokenizer
        try:
            # Try to create a subword tokenizer with the specified model path
            # The SubwordTokenizer class now handles invalid model files internally
            subword_tokenizer = SubwordTokenizer(vocab_size=args.subword_vocab_size, model_path=args.subword_model_path)

            # Check if we need to train a model
            if subword_tokenizer.sp_model is None:
                print("Training a new SentencePiece model...")
                # Create a sample training file if it doesn't exist
                train_file = "data/train.txt"
                os.makedirs(os.path.dirname(train_file), exist_ok=True)

                if not os.path.exists(train_file):
                    with open(train_file, 'w', encoding='utf-8') as f:
                        # Sample Vietnamese sentences
                        sentences = [
                            "Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói.",
                            "Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây.",
                            "Mô hình CTC đã cải thiện đáng kể độ chính xác của hệ thống nhận dạng tiếng nói.",
                            "Việt Nam có một nền văn hóa phong phú và đa dạng với nhiều dân tộc khác nhau.",
                            "Chúng tôi đang nghiên cứu các phương pháp cải thiện độ chính xác của hệ thống ASR."
                        ]
                        f.write('\n'.join(sentences))

                # Train the model
                model_prefix = args.subword_model_path[:-6]  # Remove .model extension
                subword_tokenizer.train(train_file, model_prefix)
        except Exception as e:
            print(f"Error initializing subword tokenizer: {e}")
            print("Creating a character-based fallback tokenizer")
            subword_tokenizer = SubwordTokenizer(vocab_size=args.subword_vocab_size)
            subword_tokenizer._create_fallback_model()

        if 'subword' in tokenizer_options:
            evaluator.add_tokenizer(subword_tokenizer, lm_path="data/lm/subword_lm.arpa" if os.path.exists("data/lm/subword_lm.arpa") else None)

    # Add other tokenizers based on options
    for tokenizer_type in tokenizer_options:
        if tokenizer_type == 'character':
            # Character tokenizer
            char_tokenizer = CharacterTokenizer()
            char_tokenizer.save("data/tokenizers/character_tokenizer.json")
            evaluator.add_tokenizer(char_tokenizer, lm_path="data/lm/character_lm.arpa" if os.path.exists("data/lm/character_lm.arpa") else None)
            print("Initialized character tokenizer")

        elif tokenizer_type == 'syllable':
            # Syllable tokenizer
            if not os.path.exists(args.syllable_vocab_path):
                print(f"Creating sample syllable vocabulary at {args.syllable_vocab_path}...")
                os.makedirs(os.path.dirname(args.syllable_vocab_path), exist_ok=True)
                with open(args.syllable_vocab_path, 'w', encoding='utf-8') as f:
                    # Common Vietnamese syllables
                    syllables = [
                        "xin", "chào", "tôi", "là", "người", "việt", "nam", "học", "tiếng", "anh",
                        "cảm", "ơn", "bạn", "rất", "vui", "được", "gặp", "nhà", "trường", "sinh",
                        "viên", "giáo", "dục", "công", "nghệ", "thông", "tin", "khoa", "học", "máy",
                        "tính", "phần", "mềm", "hệ", "thống", "mạng", "internet", "điện", "thoại", "di",
                        "động", "thời", "gian", "ngày", "tháng", "năm", "giờ", "phút", "giây", "sáng",
                        "trưa", "chiều", "tối", "đêm", "hôm", "nay", "mai", "qua", "kia", "mốt"
                    ]
                    f.write("\n".join(syllables))

            syllable_tokenizer = SyllableTokenizer(vocab_file=args.syllable_vocab_path)
            evaluator.add_tokenizer(syllable_tokenizer, lm_path="data/lm/syllable_lm.arpa" if os.path.exists("data/lm/syllable_lm.arpa") else None)
            print(f"Initialized syllable tokenizer with vocabulary from {args.syllable_vocab_path}")

        elif tokenizer_type == 'word':
            # Word tokenizer
            if not os.path.exists(args.word_vocab_path):
                print(f"Creating sample word vocabulary at {args.word_vocab_path}...")
                os.makedirs(os.path.dirname(args.word_vocab_path), exist_ok=True)
                with open(args.word_vocab_path, 'w', encoding='utf-8') as f:
                    # Common Vietnamese words (multi-syllable)
                    words = [
                        "xin chào", "cảm ơn", "việt nam", "học sinh", "sinh viên", "giáo dục",
                        "công nghệ", "thông tin", "khoa học", "máy tính", "phần mềm", "hệ thống",
                        "mạng internet", "điện thoại", "di động", "thời gian", "ngày tháng",
                        "năm học", "giờ phút", "sáng sớm", "buổi trưa", "buổi chiều", "buổi tối",
                        "đêm khuya", "hôm nay", "ngày mai", "hôm qua", "ngày kia", "tuần trước",
                        "tháng sau", "năm ngoái", "thế kỷ", "quốc gia", "dân tộc", "con người",
                        "gia đình", "bạn bè", "người thân", "anh chị", "cha mẹ", "ông bà"
                    ]
                    f.write("\n".join(words))

            if subword_tokenizer is None:
                print("Warning: Word tokenizer requires subword tokenizer for OOV handling. Creating a character-based fallback tokenizer.")
                subword_tokenizer = SubwordTokenizer(vocab_size=args.subword_vocab_size)
                subword_tokenizer._create_fallback_model()

            word_tokenizer = WordTokenizer(vocab_file=args.word_vocab_path, subword_tokenizer=subword_tokenizer)
            evaluator.add_tokenizer(word_tokenizer, lm_path="data/lm/word_lm.arpa" if os.path.exists("data/lm/word_lm.arpa") else None)
            print(f"Initialized word tokenizer with vocabulary from {args.word_vocab_path}")

    if not evaluator.tokenizers:
        print("Error: No valid tokenizers specified. Please use --tokenizers with 'character', 'subword', 'syllable', 'word', or 'all'")
        return

    # Create sample audio and transcript files if they don't exist
    if not os.listdir(args.audio_dir) or not os.listdir(args.transcript_dir):
        print("Creating sample data...")
        create_sample_data(args.audio_dir, args.transcript_dir)

    # Prepare test data
    print("Preparing test data...")
    test_data = evaluator.prepare_test_data(num_samples=args.num_samples)

    if not test_data:
        print("No test data found. Please add audio files and transcripts.")
        return

    # Evaluate
    print("Evaluating CTC decoders...")
    results = evaluator.evaluate(
        test_data,
        beam_size=args.beam_size,
        alpha=args.alpha,
        beta=args.beta
    )

    # Generate report
    print("Generating report...")
    report_dir = evaluator.generate_report(results)

    print(f"\nEvaluation complete. Results saved to {report_dir}")
    print(f"Open {os.path.join(report_dir, 'report.html')} to view the full report.")

def create_sample_data(audio_dir: str, transcript_dir: str, num_samples: int = 5):
    """Create sample audio and transcript files for testing."""
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    # Sample Vietnamese sentences
    sentences = [
        "Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói.",
        "Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây.",
        "Mô hình CTC đã cải thiện đáng kể độ chính xác của hệ thống nhận dạng tiếng nói.",
        "Việt Nam có một nền văn hóa phong phú và đa dạng với nhiều dân tộc khác nhau.",
        "Chúng tôi đang nghiên cứu các phương pháp cải thiện độ chính xác của hệ thống ASR."
    ]

    # Create sample audio files (1-second silence) and transcripts
    for i in range(min(num_samples, len(sentences))):
        # Create a simple audio file (1 second of silence at 16kHz)
        sample_rate = 16000

        # Save audio file
        audio_path = os.path.join(audio_dir, f"sample_{i+1:02d}.wav")
        try:
            waveform = torch.zeros(1, sample_rate)  # 1 second of silence
            torchaudio.save(audio_path, waveform, sample_rate)
        except Exception as e:
            print(f"Error saving audio file {audio_path}: {e}")
            print("Creating an empty file as a fallback")
            # Create an empty file as a fallback
            with open(audio_path, 'wb') as f:
                f.write(b'')

        # Save transcript
        transcript_path = os.path.join(transcript_dir, f"sample_{i+1:02d}.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(sentences[i])

    print(f"Created {min(num_samples, len(sentences))} sample audio files and transcripts.")

if __name__ == "__main__":
    main()
