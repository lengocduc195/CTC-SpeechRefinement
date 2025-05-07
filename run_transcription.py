"""
Script to run speech transcription using the new package structure.
"""

import argparse
import logging
import os
from pathlib import Path

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ctc_speech_refinement.apps.transcription.transcribe import main as transcription_main

if __name__ == "__main__":
    transcription_main()
