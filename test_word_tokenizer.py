#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the WordTokenizer with the new Vietnamese word list.
"""

import os
from ctc_eval import WordTokenizer, SubwordTokenizer

def main():
    """Main function to test the WordTokenizer."""
    # Initialize the SubwordTokenizer (needed for WordTokenizer's OOV handling)
    subword_tokenizer = SubwordTokenizer(vocab_size=5000)
    subword_tokenizer._create_fallback_model()
    
    # Initialize the WordTokenizer with the new word list
    word_vocab_path = "data/tokenizers/vietnamese_words.txt"
    word_tokenizer = WordTokenizer(vocab_file=word_vocab_path, subword_tokenizer=subword_tokenizer)
    
    # Print some statistics
    print(f"Loaded word vocabulary with {word_tokenizer.vocab_size} words")
    
    # Test some sample Vietnamese sentences
    test_sentences = [
        "Xin chào, tôi đang thử nghiệm hệ thống nhận dạng tiếng nói.",
        "Công nghệ nhận dạng tiếng nói đã phát triển rất nhanh trong những năm gần đây.",
        "Mô hình CTC đã cải thiện đáng kể độ chính xác của hệ thống nhận dạng tiếng nói.",
        "Việt Nam có một nền văn hóa phong phú và đa dạng với nhiều dân tộc khác nhau.",
        "Chúng tôi đang nghiên cứu các phương pháp cải thiện độ chính xác của hệ thống ASR."
    ]
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nTest sentence {i+1}: {sentence}")
        
        # Encode the sentence
        token_ids = word_tokenizer.encode(sentence)
        print(f"Encoded token IDs: {token_ids}")
        
        # Decode the token IDs
        decoded_text = word_tokenizer.decode(token_ids)
        print(f"Decoded text: {decoded_text}")
        
        # Check for OOV words
        words = sentence.lower().split()
        oov_words = []
        for word in words:
            if word not in word_tokenizer.token_to_id:
                oov_words.append(word)
        
        if oov_words:
            print(f"OOV words: {oov_words}")
        else:
            print("No OOV words found")

if __name__ == "__main__":
    main()
