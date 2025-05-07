"""
Command-line script for running audio exploratory data analysis (EDA).
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any

from src.eda.audio_eda import analyze_audio_file, batch_analyze_audio_files, analyze_directory
from src.utils.file_utils import get_audio_files
from config.config import TEST1_DIR, RESULTS_DIR

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
        help="Input audio file or directory containing audio files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(str(RESULTS_DIR), "eda"),
        help="Directory to save analysis results and plots"
    )

    # Basic preprocessing options
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize the audio"
    )

    parser.add_argument(
        "--remove_silence",
        action="store_true",
        help="Whether to remove silence from the audio"
    )

    # Advanced preprocessing options
    parser.add_argument(
        "--trim_start",
        type=float,
        default=0.0,
        help="Start time for trimming in seconds"
    )

    parser.add_argument(
        "--trim_end",
        type=float,
        default=None,
        help="End time for trimming in seconds"
    )

    parser.add_argument(
        "--fade_in",
        type=float,
        default=0.0,
        help="Fade-in time in seconds"
    )

    parser.add_argument(
        "--fade_out",
        type=float,
        default=0.0,
        help="Fade-out time in seconds"
    )

    parser.add_argument(
        "--filter_type",
        type=str,
        choices=["lowpass", "highpass", "bandpass", "bandstop"],
        default=None,
        help="Type of filter to apply"
    )

    parser.add_argument(
        "--cutoff_freq",
        type=float,
        default=1000.0,
        help="Cutoff frequency for the filter in Hz (for bandpass/bandstop, use --cutoff_low and --cutoff_high)"
    )

    parser.add_argument(
        "--cutoff_low",
        type=float,
        default=500.0,
        help="Lower cutoff frequency for bandpass/bandstop filter in Hz"
    )

    parser.add_argument(
        "--cutoff_high",
        type=float,
        default=2000.0,
        help="Upper cutoff frequency for bandpass/bandstop filter in Hz"
    )

    parser.add_argument(
        "--add_noise",
        type=float,
        default=None,
        help="Level of noise to add (standard deviation)"
    )

    # Other options
    parser.add_argument(
        "--single_file",
        action="store_true",
        help="Treat input as a single file even if it's a directory"
    )

    return parser.parse_args()

def main():
    """Main function for audio EDA."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Prepare cutoff frequency for bandpass/bandstop filters
    cutoff_freq = args.cutoff_freq
    if args.filter_type in ["bandpass", "bandstop"]:
        cutoff_freq = (args.cutoff_low, args.cutoff_high)

    if args.single_file or os.path.isfile(args.input):
        # Analyze a single file
        logger.info(f"Analyzing single audio file: {args.input}")
        results = analyze_audio_file(
            args.input,
            args.output_dir,
            args.normalize,
            args.remove_silence,
            args.trim_start,
            args.trim_end,
            args.fade_in,
            args.fade_out,
            args.filter_type,
            cutoff_freq,
            args.add_noise
        )
        logger.info(f"Analysis completed for {args.input}")
    else:
        # Analyze a directory
        logger.info(f"Analyzing audio files in directory: {args.input}")
        results = analyze_directory(
            args.input,
            args.output_dir,
            args.normalize,
            args.remove_silence,
            args.trim_start,
            args.trim_end,
            args.fade_in,
            args.fade_out,
            args.filter_type,
            cutoff_freq,
            args.add_noise
        )
        logger.info(f"Analysis completed for {len(results)} files in {args.input}")

    elapsed_time = time.time() - start_time
    logger.info(f"Total analysis time: {elapsed_time:.2f} seconds")

    # Print summary
    if isinstance(results, dict) and not args.single_file:
        logger.info(f"Analysis summary:")
        logger.info(f"  Number of files analyzed: {len(results)}")

        if results:
            # Calculate average duration
            avg_duration = sum(result.get("duration", 0) for result in results.values()) / len(results)
            logger.info(f"  Average duration: {avg_duration:.2f} seconds")

            # Count files with anomalies
            files_with_amplitude_anomalies = sum(
                1 for result in results.values()
                if len(result.get("anomalies", {}).get("amplitude_anomalies", [])) > 0
            )

            files_with_spectral_anomalies = sum(
                1 for result in results.values()
                if len(result.get("anomalies", {}).get("spectral_anomalies", [])) > 0
            )

            logger.info(f"  Files with amplitude anomalies: {files_with_amplitude_anomalies}")
            logger.info(f"  Files with spectral anomalies: {files_with_spectral_anomalies}")

    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
