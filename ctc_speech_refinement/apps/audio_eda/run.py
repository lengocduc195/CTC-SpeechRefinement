"""
Module for running audio exploratory data analysis (EDA).
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any

from ctc_speech_refinement.core.eda.audio_eda import analyze_audio_file, batch_analyze_audio_files, analyze_directory
from ctc_speech_refinement.core.utils.file_utils import get_audio_files
from ctc_speech_refinement.config.config import TEST1_DIR, RESULTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_eda.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Audio Exploratory Data Analysis")
    
    parser.add_argument(
        "--input", 
        type=str, 
        default=str(TEST1_DIR),
        help="Path to audio file or directory"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.join(str(RESULTS_DIR), "eda"),
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--normalize", 
        action="store_true",
        help="Whether to normalize audio"
    )
    
    parser.add_argument(
        "--remove_silence", 
        action="store_true",
        help="Whether to remove silence from audio"
    )
    
    parser.add_argument(
        "--plot_format", 
        type=str, 
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for saving plots"
    )
    
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=100,
        help="DPI for saving plots"
    )
    
    parser.add_argument(
        "--save_audio", 
        action="store_true",
        help="Whether to save preprocessed audio files"
    )
    
    return parser.parse_args()

def main():
    """Main function for audio EDA."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Analyze a single file
        logger.info(f"Analyzing audio file: {input_path}")
        
        start_time = time.time()
        results = analyze_audio_file(
            str(input_path),
            output_dir=args.output_dir,
            normalize=args.normalize,
            remove_silence=args.remove_silence,
            plot_format=args.plot_format,
            dpi=args.dpi,
            save_audio=args.save_audio
        )
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        
        # Print summary
        print("\n===== AUDIO ANALYSIS SUMMARY =====")
        print(f"File: {input_path}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Sample rate: {results['sample_rate']} Hz")
        print(f"Mean amplitude: {results['mean_amplitude']:.6f}")
        print(f"RMS: {results['rms']:.6f}")
        print(f"Results saved to: {args.output_dir}")
        
    elif input_path.is_dir():
        # Analyze a directory
        logger.info(f"Analyzing audio files in directory: {input_path}")
        
        start_time = time.time()
        results = analyze_directory(
            str(input_path),
            output_dir=args.output_dir,
            normalize=args.normalize,
            remove_silence=args.remove_silence,
            plot_format=args.plot_format,
            dpi=args.dpi,
            save_audio=args.save_audio
        )
        logger.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        
        # Print summary
        print("\n===== AUDIO ANALYSIS SUMMARY =====")
        print(f"Directory: {input_path}")
        print(f"Number of files analyzed: {len(results)}")
        print(f"Results saved to: {args.output_dir}")
        
    else:
        logger.error(f"Input path not found: {input_path}")
        return
    
    logger.info("Audio EDA completed successfully")

if __name__ == "__main__":
    main()
