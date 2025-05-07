"""
Script to run the audio preprocessing UI.
"""

import tkinter as tk
import argparse
import logging
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.preprocessing_ui import PreprocessingUI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the preprocessing UI."""
    parser = argparse.ArgumentParser(description="Audio Preprocessing UI")
    parser.add_argument("--language", "-l", choices=["en", "vi"], default="vi", help="UI language (en or vi)")
    args = parser.parse_args()
    
    root = tk.Tk()
    app = PreprocessingUI(root, language=args.language)
    root.mainloop()

if __name__ == "__main__":
    main()
