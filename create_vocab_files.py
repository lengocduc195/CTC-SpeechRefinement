#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create vocabulary files for Vietnamese syllable and word tokenizers.

This script creates vocabulary files for syllable-level and word-level
tokenization of Vietnamese text. These files can be used with the
CTC decoder evaluation script.

Usage:
    python create_vocab_files.py --syllable_file data/tokenizers/vietnamese_syllables.txt --word_file data/tokenizers/vietnamese_words.txt

Author: AI Assistant
"""

import os
import argparse
from typing import List, Set, Tuple

def extract_syllables_and_words(text: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract syllables and words from Vietnamese text.
    
    Args:
        text: Vietnamese text
        
    Returns:
        Tuple of (syllables, words) sets
    """
    # Normalize text
    text = text.lower()
    
    # Split into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    syllables = set()
    words = set()
    
    for sentence in sentences:
        # In Vietnamese, syllables are separated by spaces
        sentence_syllables = [s for s in sentence.split() if s]
        syllables.update(sentence_syllables)
        
        # Words can be single syllables or multiple syllables
        # This is a simplified approach - proper word segmentation is more complex
        i = 0
        while i < len(sentence_syllables):
            # Single syllable word
            words.add(sentence_syllables[i])
            
            # Try multi-syllable words (up to 3 syllables)
            for j in range(1, min(3, len(sentence_syllables) - i)):
                multi_syllable = ' '.join(sentence_syllables[i:i+j+1])
                words.add(multi_syllable)
            
            i += 1
    
    return syllables, words

def save_vocabulary(items: List[str], output_file: str) -> None:
    """
    Save vocabulary items to a file.
    
    Args:
        items: List of vocabulary items
        output_file: Path to save the vocabulary
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sorted(items):
            f.write(item + '\n')
    
    print(f"Saved {len(items)} items to {output_file}")

def create_sample_vocabulary() -> Tuple[List[str], List[str]]:
    """
    Create sample Vietnamese syllable and word vocabularies.
    
    Returns:
        Tuple of (syllables, words) lists
    """
    # Common Vietnamese syllables
    syllables = [
        "xin", "chào", "tôi", "là", "người", "việt", "nam", "học", "tiếng", "anh",
        "cảm", "ơn", "bạn", "rất", "vui", "được", "gặp", "nhà", "trường", "sinh",
        "viên", "giáo", "dục", "công", "nghệ", "thông", "tin", "khoa", "học", "máy",
        "tính", "phần", "mềm", "hệ", "thống", "mạng", "internet", "điện", "thoại", "di",
        "động", "thời", "gian", "ngày", "tháng", "năm", "giờ", "phút", "giây", "sáng",
        "trưa", "chiều", "tối", "đêm", "hôm", "nay", "mai", "qua", "kia", "mốt",
        "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
        "trăm", "nghìn", "triệu", "tỷ", "và", "hoặc", "nhưng", "vì", "nên", "do",
        "bởi", "nếu", "thì", "khi", "lúc", "sau", "trước", "trong", "ngoài", "trên",
        "dưới", "đây", "đó", "kia", "này", "ấy", "ai", "gì", "nào", "sao"
    ]
    
    # Common Vietnamese words (multi-syllable)
    words = [
        "xin chào", "cảm ơn", "việt nam", "học sinh", "sinh viên", "giáo dục",
        "công nghệ", "thông tin", "khoa học", "máy tính", "phần mềm", "hệ thống",
        "mạng internet", "điện thoại", "di động", "thời gian", "ngày tháng",
        "năm học", "giờ phút", "sáng sớm", "buổi trưa", "buổi chiều", "buổi tối",
        "đêm khuya", "hôm nay", "ngày mai", "hôm qua", "ngày kia", "tuần trước",
        "tháng sau", "năm ngoái", "thế kỷ", "quốc gia", "dân tộc", "con người",
        "gia đình", "bạn bè", "người thân", "anh chị", "cha mẹ", "ông bà",
        "trường học", "bệnh viện", "nhà hàng", "siêu thị", "chợ búa", "công viên",
        "đường phố", "thành phố", "nông thôn", "miền núi", "đồng bằng", "biển cả",
        "sông ngòi", "núi non", "rừng rậm", "sa mạc", "đại dương", "hành tinh",
        "vũ trụ", "mặt trời", "mặt trăng", "ngôi sao", "thiên hà", "trái đất"
    ]
    
    # Add single syllables to words
    words.extend(syllables)
    
    return syllables, words

def create_vocabulary_from_text(input_file: str, syllable_file: str, word_file: str) -> None:
    """
    Create vocabulary files from Vietnamese text.
    
    Args:
        input_file: Path to the input text file
        syllable_file: Path to save the syllable vocabulary
        word_file: Path to save the word vocabulary
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    syllables, words = extract_syllables_and_words(text)
    
    save_vocabulary(list(syllables), syllable_file)
    save_vocabulary(list(words), word_file)

def main():
    """Main function to create vocabulary files."""
    parser = argparse.ArgumentParser(description="Create vocabulary files for Vietnamese tokenizers")
    parser.add_argument("--input_file", type=str, default="",
                       help="Path to the input text file")
    parser.add_argument("--syllable_file", type=str, default="data/tokenizers/vietnamese_syllables.txt",
                       help="Path to save the syllable vocabulary")
    parser.add_argument("--word_file", type=str, default="data/tokenizers/vietnamese_words.txt",
                       help="Path to save the word vocabulary")
    parser.add_argument("--use_sample", action="store_true",
                       help="Use sample vocabulary instead of extracting from text")
    args = parser.parse_args()
    
    if args.use_sample or not args.input_file:
        print("Creating sample vocabulary files...")
        syllables, words = create_sample_vocabulary()
        save_vocabulary(syllables, args.syllable_file)
        save_vocabulary(words, args.word_file)
    else:
        print(f"Creating vocabulary files from {args.input_file}...")
        create_vocabulary_from_text(args.input_file, args.syllable_file, args.word_file)
    
    print("\nTo use these vocabulary files in the CTC decoder evaluation, run:")
    print(f"python ctc_eval.py --tokenizers syllable,word --syllable_vocab_path {args.syllable_file} --word_vocab_path {args.word_file}")

if __name__ == "__main__":
    main()
