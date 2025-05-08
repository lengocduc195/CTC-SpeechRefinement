"""
Script to run error analysis on speech recognition results.
"""

import argparse
import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from ctc_speech_refinement.core.utils.file_utils import load_transcriptions
from ctc_speech_refinement.core.eda.error_analysis import ErrorAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Error Analysis for Speech Recognition")

    parser.add_argument(
        "--reference_dir",
        type=str,
        required=True,
        help="Directory containing reference transcriptions"
    )

    parser.add_argument(
        "--hypothesis_dir",
        type=str,
        required=True,
        help="Directory containing hypothesis (recognized) transcriptions"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/error_analysis",
        help="Directory to save error analysis results"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="Language of the transcriptions"
    )

    return parser.parse_args()

def main():
    """Main function for error analysis."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load transcriptions
    logger.info(f"Loading reference transcriptions from {args.reference_dir}")
    reference_transcripts = load_transcriptions(args.reference_dir)

    logger.info(f"Loading hypothesis transcriptions from {args.hypothesis_dir}")
    hypothesis_transcripts = load_transcriptions(args.hypothesis_dir)

    if not reference_transcripts:
        logger.error(f"No reference transcriptions found in {args.reference_dir}")
        return

    if not hypothesis_transcripts:
        logger.error(f"No hypothesis transcriptions found in {args.hypothesis_dir}")
        return

    logger.info(f"Loaded {len(reference_transcripts)} reference transcriptions and {len(hypothesis_transcripts)} hypothesis transcriptions")

    # Initialize error analyzer
    analyzer = ErrorAnalyzer(
        reference_transcripts=reference_transcripts,
        hypothesis_transcripts=hypothesis_transcripts,
        language=args.language
    )

    # Analyze errors
    start_time = time.time()
    logger.info("Analyzing errors...")
    analyzer.analyze_all_transcripts()
    logger.info(f"Error analysis completed in {time.time() - start_time:.2f} seconds")

    # Generate error report
    logger.info("Generating error report...")
    report = analyzer.generate_error_report(args.output_dir)

    # Print summary
    print("\n===== ERROR ANALYSIS SUMMARY =====")
    print(f"Total files analyzed: {report['summary']['total_files']}")
    print(f"Average WER: {report['summary']['avg_wer']:.4f}")
    print(f"Average CER: {report['summary']['avg_cer']:.4f}")
    print(f"Total substitutions: {report['summary']['substitutions']}")
    print(f"Total deletions: {report['summary']['deletions']}")
    print(f"Total insertions: {report['summary']['insertions']}")

    print("\nMost common substitution errors:")
    for i, sub in enumerate(report['detailed_errors']['substitutions'][:5], 1):
        print(f"{i}. '{sub['reference']}' -> '{sub['hypothesis']}' (Count: {sub['count']})")

    print("\nMost common deletion errors:")
    for i, deletion in enumerate(report['detailed_errors']['deletions'][:5], 1):
        print(f"{i}. '{deletion['word']}' (Count: {deletion['count']})")

    print("\nMost common insertion errors:")
    for i, insertion in enumerate(report['detailed_errors']['insertions'][:5], 1):
        print(f"{i}. '{insertion['word']}' (Count: {insertion['count']})")

    print(f"\nDetailed report saved to {os.path.join(args.output_dir, 'error_analysis_report.json')}")
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
