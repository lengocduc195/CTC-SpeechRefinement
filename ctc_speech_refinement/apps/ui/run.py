"""
Module for running the preprocessing UI.
"""

import argparse
import logging
import os
from pathlib import Path

from ctc_speech_refinement.core.ui.preprocessing_ui import run_preprocessing_ui

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio Preprocessing UI")
    
    parser.add_argument(
        "--language", 
        type=str, 
        default="en",
        help="Language for the UI"
    )
    
    return parser.parse_args()

def main():
    """Main function for preprocessing UI."""
    args = parse_args()
    
    logger.info(f"Starting preprocessing UI with language: {args.language}")
    
    # Run the preprocessing UI
    run_preprocessing_ui(language=args.language)
    
    logger.info("Preprocessing UI closed")

if __name__ == "__main__":
    main()
