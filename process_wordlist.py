#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Vietnamese word list to create a suitable tokenizer vocabulary.

This script processes a Vietnamese word list from the duyet/vietnamese-wordlist
repository and creates a filtered and formatted word list suitable for use
with the CTC decoder's word tokenizer.

Usage:
    python process_wordlist.py --input vietnamese_wordlist.txt --output data/tokenizers/vietnamese_words.txt
"""

import os
import argparse
import re
from typing import List, Set

def is_valid_vietnamese_word(word: str) -> bool:
    """
    Check if a word is a valid Vietnamese word.

    Args:
        word: Word to check

    Returns:
        True if the word is valid, False otherwise
    """
    # Skip empty words
    if not word.strip():
        return False

    # Skip words with non-Vietnamese characters
    # Vietnamese characters include a-z, A-Z, Vietnamese diacritics, and spaces
    vietnamese_pattern = r'^[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ\s]+$'
    if not re.match(vietnamese_pattern, word):
        return False

    return True

def process_word_list(input_file: str, output_file: str, max_words: int = 10000) -> None:
    """
    Process a Vietnamese word list and save it to a file.

    Args:
        input_file: Path to the input word list
        output_file: Path to save the processed word list
        max_words: Maximum number of words to include
    """
    # Try different encodings to read the file
    encodings = ['utf-8', 'latin-1', 'cp1252']
    words = []

    for encoding in encodings:
        try:
            with open(input_file, 'r', encoding=encoding) as f:
                words = [line.strip() for line in f]
            print(f"Successfully read file with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to read file with encoding: {encoding}")

    if not words:
        print("Failed to read file with any encoding. Creating a basic word list.")
        words = ["xin", "chào", "việt", "nam"]

    # Create a new list with common Vietnamese words
    common_words = [
        "xin chào", "cảm ơn", "việt nam", "học sinh", "sinh viên", "giáo dục",
        "công nghệ", "thông tin", "khoa học", "máy tính", "phần mềm", "hệ thống",
        "mạng internet", "điện thoại", "di động", "thời gian", "ngày tháng",
        "năm học", "giờ phút", "sáng sớm", "buổi trưa", "buổi chiều", "buổi tối",
        "đêm khuya", "hôm nay", "ngày mai", "hôm qua", "ngày kia", "tuần trước",
        "tháng sau", "năm ngoái", "thế kỷ", "quốc gia", "dân tộc", "con người",
        "gia đình", "bạn bè", "người thân", "anh chị", "cha mẹ", "ông bà",
        "a", "ạ", "á", "à", "ả", "ác", "ai", "am", "an", "anh", "ao", "áo", "áp",
        "ắt", "âm", "ấm", "ân", "ấn", "âu", "ấy", "ba", "bà", "bác", "bạc", "bài",
        "bán", "bạn", "bao", "bát", "bay", "bày", "bảy", "bắc", "bắt", "bằng", "bận",
        "bất", "bây", "bấy", "bên", "bền", "bể", "bệnh", "bí", "bị", "biên", "biển",
        "biết", "biếu", "binh", "bình", "bo", "bó", "bỏ", "bọc", "bói", "bom", "bóng",
        "bốn", "bông", "bố", "bổ", "bộ", "bờ", "bởi", "bới", "bức", "bưu", "bữa", "bực",
        "bước", "bướm", "bướu", "ca", "cá", "các", "cách", "cái", "cam", "cảm", "can",
        "cán", "canh", "cánh", "cao", "cào", "cáo", "cạo", "cay", "cày", "cạy", "cắm",
        "cắn", "cắt", "căn", "cân", "cần", "cật", "cây", "cha", "chà", "chác", "chạc",
        "chai", "chải", "chàm", "chán", "chạn", "chăm", "chăn", "chăng", "chặng", "chặt",
        "chân", "chấn", "chất", "chầu", "chây", "chậm", "chậu", "chèn", "chẻ", "chém",
        "chén", "chèo", "chết", "chê", "chế", "chễm", "chện", "chi", "chí", "chìa", "chìm",
        "chín", "chỉ", "chị", "chịu", "cho", "chó", "chọc", "chọi", "chọn", "chống", "chọt",
        "chơi", "chở", "chờ", "chợ", "chớ", "chợt", "chu", "chú", "chùa", "chúa", "chúc",
        "chui", "chúm", "chung", "chùng", "chúng", "chuyên", "chuyển", "chuyện", "chữ",
        "chữa", "chức", "chưa", "chửa", "chứa", "chức", "chứng", "chước", "chương", "chước",
        "có", "cô", "cổ", "cố", "cộ", "cốc", "cộc", "coi", "cõi", "con", "còn", "cón",
        "cọn", "cong", "cõng", "công", "cộng", "cót", "cô", "cổ", "cố", "cốc", "cộc",
        "coi", "cõi", "con", "còn", "cón", "cọn", "cong", "cõng", "công", "cộng", "cót",
        "cô", "cổ", "cố", "cốc", "cộc", "coi", "cõi", "con", "còn", "cón", "cọn", "cong",
        "cõng", "công", "cộng", "cót", "cô", "cổ", "cố", "cốc", "cộc", "coi", "cõi", "con",
        "còn", "cón", "cọn", "cong", "cõng", "công", "cộng", "cót", "cô", "cổ", "cố", "cốc",
        "cộc", "coi", "cõi", "con", "còn", "cón", "cọn", "cong", "cõng", "công", "cộng", "cót"
    ]

    # Add single syllable words from the original list
    for word in words:
        if " " not in word and word not in common_words:
            common_words.append(word)

    # Limit the number of words
    if max_words > 0 and len(common_words) > max_words:
        common_words = common_words[:max_words]

    # Save the processed word list
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(common_words))

    print(f"Processed {len(words)} words, saved {len(common_words)} words to {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process Vietnamese word list")
    parser.add_argument("--input", type=str, default="vietnamese_wordlist.txt",
                       help="Path to the input word list")
    parser.add_argument("--output", type=str, default="data/tokenizers/vietnamese_words.txt",
                       help="Path to save the processed word list")
    parser.add_argument("--max_words", type=int, default=10000,
                       help="Maximum number of words to include (0 for all)")

    args = parser.parse_args()

    process_word_list(args.input, args.output, args.max_words)

if __name__ == "__main__":
    main()
