#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a SentencePiece BPE model for Vietnamese tokenization.

This script trains a Byte-Pair Encoding (BPE) model using SentencePiece
for Vietnamese text. It can be used to create a subword tokenizer for
the CTC decoder evaluation.

Usage:
    python train_bpe_model.py --input_file data/train.txt --vocab_size 5000 --model_prefix data/tokenizers/vietnamese_bpe_5000

Author: AI Assistant
"""

import os
import argparse
import sentencepiece as spm
from typing import List, Optional

def prepare_training_data(sentences: List[str], output_file: str) -> str:
    """
    Prepare training data for SentencePiece.
    
    Args:
        sentences: List of Vietnamese sentences
        output_file: Path to save the training data
        
    Returns:
        Path to the training data file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence.strip() + '\n')
    
    return output_file

def train_bpe_model(input_file: str, model_prefix: str, vocab_size: int = 5000,
                   character_coverage: float = 1.0) -> None:
    """
    Train a SentencePiece BPE model.
    
    Args:
        input_file: Path to the training data file
        model_prefix: Prefix for the output model files
        vocab_size: Size of the vocabulary
        character_coverage: Character coverage (1.0 for Vietnamese)
    """
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    
    # Train the model
    spm.SentencePieceTrainer.Train(
        f'--input={input_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--character_coverage={character_coverage} '
        f'--model_type=bpe '
        f'--pad_id=-1 '
        f'--unk_id=0 '
        f'--bos_id=1 '
        f'--eos_id=2 '
        f'--normalization_rule_name=identity'
    )
    
    print(f"Model trained and saved to {model_prefix}.model and {model_prefix}.vocab")

def generate_sample_vietnamese_text(output_file: str, num_sentences: int = 1000) -> str:
    """
    Generate sample Vietnamese text for training.
    
    Args:
        output_file: Path to save the sample text
        num_sentences: Number of sentences to generate
        
    Returns:
        Path to the sample text file
    """
    # Sample Vietnamese sentences
    sentences = [
        "Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói.",
        "Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây.",
        "Mô hình CTC đã cải thiện đáng kể độ chính xác của hệ thống nhận dạng tiếng nói.",
        "Việt Nam có một nền văn hóa phong phú và đa dạng với nhiều dân tộc khác nhau.",
        "Chúng tôi đang nghiên cứu các phương pháp cải thiện độ chính xác của hệ thống ASR.",
        "Hà Nội là thủ đô của Việt Nam, một thành phố với lịch sử hơn nghìn năm văn hiến.",
        "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất của Việt Nam.",
        "Tiếng Việt là một ngôn ngữ có thanh điệu với sáu thanh khác nhau.",
        "Việc xử lý ngôn ngữ tự nhiên cho tiếng Việt đặt ra nhiều thách thức đặc biệt.",
        "Các mô hình ngôn ngữ đóng vai trò quan trọng trong việc cải thiện độ chính xác của ASR.",
        "Tokenization là một bước quan trọng trong quá trình xử lý ngôn ngữ tự nhiên.",
        "Byte-Pair Encoding là một phương pháp tokenization hiệu quả cho nhiều ngôn ngữ.",
        "Connectionist Temporal Classification giúp giải quyết vấn đề alignment trong ASR.",
        "Greedy decoding là phương pháp đơn giản nhất để giải mã đầu ra của mô hình CTC.",
        "Beam search decoding kết hợp với mô hình ngôn ngữ cho kết quả tốt hơn greedy decoding.",
        "Đánh giá hệ thống ASR thường sử dụng các metric như WER, CER và SER.",
        "Character Error Rate đo lường tỷ lệ lỗi ở cấp độ ký tự.",
        "Word Error Rate đo lường tỷ lệ lỗi ở cấp độ từ.",
        "Syllable Error Rate đặc biệt quan trọng đối với tiếng Việt.",
        "Việc lựa chọn mức độ tokenization phụ thuộc vào yêu cầu cụ thể của ứng dụng."
    ]
    
    # Repeat sentences to get the desired number
    sentences = sentences * (num_sentences // len(sentences) + 1)
    sentences = sentences[:num_sentences]
    
    return prepare_training_data(sentences, output_file)

def main():
    """Main function to train a BPE model."""
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE model for Vietnamese")
    parser.add_argument("--input_file", type=str, default="",
                       help="Path to the training data file")
    parser.add_argument("--vocab_size", type=int, default=5000,
                       help="Size of the vocabulary")
    parser.add_argument("--model_prefix", type=str, default="data/tokenizers/vietnamese_bpe_5000",
                       help="Prefix for the output model files")
    parser.add_argument("--generate_sample", action="store_true",
                       help="Generate sample Vietnamese text for training")
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # Generate sample text if requested or if no input file is provided
    if args.generate_sample or not input_file:
        print("Generating sample Vietnamese text...")
        input_file = generate_sample_vietnamese_text("data/train.txt")
        print(f"Sample text generated at {input_file}")
    
    # Train the model
    print(f"Training BPE model with vocabulary size {args.vocab_size}...")
    train_bpe_model(input_file, args.model_prefix, args.vocab_size)
    
    print("\nTo use this model in the CTC decoder evaluation, run:")
    print(f"python ctc_eval.py --tokenizers subword --subword_model_path {args.model_prefix}.model")

if __name__ == "__main__":
    main()
