"""
Script to fix import paths in Python modules.
"""

import os
import re
import glob

def fix_imports_in_file(file_path):
    """
    Fix import paths in a Python file.

    Args:
        file_path: Path to the Python file.
    """
    print(f"Fixing imports in {file_path}")

    try:
        # Try to read the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace src.* imports with ctc_speech_refinement.core.*
        content = re.sub(r'from src\.', 'from ctc_speech_refinement.core.', content)
        content = re.sub(r'import src\.', 'import ctc_speech_refinement.core.', content)

        # Replace config.* imports with ctc_speech_refinement.config.*
        content = re.sub(r'from config\.', 'from ctc_speech_refinement.config.', content)
        content = re.sub(r'import config\.', 'import ctc_speech_refinement.config.', content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except UnicodeDecodeError:
        print(f"  Skipping {file_path} - not a text file or not UTF-8 encoded")

def fix_imports_in_directory(directory):
    """
    Fix import paths in all Python files in a directory and its subdirectories.

    Args:
        directory: Directory to search for Python files.
    """
    print(f"Fixing imports in directory: {directory}")

    # Find all Python files
    python_files = glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)

    for file_path in python_files:
        fix_imports_in_file(file_path)

if __name__ == "__main__":
    # Fix imports in the core directory
    fix_imports_in_directory('ctc_speech_refinement/core')
