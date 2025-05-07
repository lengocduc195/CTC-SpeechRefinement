"""
Script to run the preprocessing UI using the new package structure.
"""

import argparse
import logging
import os
from pathlib import Path

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ctc_speech_refinement.apps.ui.run import main as preprocessing_ui_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Preprocessing UI")
    parser.add_argument("--language", "-l", choices=["en", "vi"], default="vi", help="UI language (en or vi)")
    args = parser.parse_args()

    preprocessing_ui_main(language=args.language)
