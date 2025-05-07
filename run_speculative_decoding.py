"""
Script to run speculative decoding with CTC-Drafter and CR-CTC using the new package structure.
"""

import argparse
import logging
import os
from pathlib import Path

# Add the project root directory to the Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ctc_speech_refinement.apps.speculative_decoding.run import main as speculative_decoding_main

if __name__ == "__main__":
    speculative_decoding_main()
