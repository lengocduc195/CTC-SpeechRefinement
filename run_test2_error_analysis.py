"""
Script to run error analysis on test2 speech recognition results.
"""

import argparse
import logging
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from src.utils.file_utils import load_transcriptions
from src.eda.error_analysis import ErrorAnalyzer
from src.eda.error_visualization import ErrorVisualizer
from src.eda.error_improvement import ErrorImprover

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test2_error_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Error Analysis for test2 Speech Recognition")

    parser.add_argument(
        "--reference_dir",
        type=str,
        default="data/test2/reference",
        help="Directory containing reference transcriptions for test2"
    )

    parser.add_argument(
        "--hypothesis_dir",
        type=str,
        default="transcripts/test2",
        help="Directory containing hypothesis (recognized) transcriptions for test2"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/test2/error_analysis",
        help="Directory to save error analysis results"
    )

    parser.add_argument(
        "--skip_visualizations",
        action="store_true",
        help="Skip generating visualizations"
    )

    return parser.parse_args()

def main():
    """Main function for test2 error analysis."""
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

    # Step 1: Initialize error analyzer and analyze errors
    analyzer = ErrorAnalyzer(
        reference_transcripts=reference_transcripts,
        hypothesis_transcripts=hypothesis_transcripts
    )

    start_time = time.time()
    logger.info("Step 1: Analyzing errors in test2 data...")
    analyzer.analyze_all_transcripts()
    logger.info(f"Error analysis completed in {time.time() - start_time:.2f} seconds")

    # Generate error report
    logger.info("Generating error report...")
    report = analyzer.generate_error_report(args.output_dir)

    # Step 2: Create advanced visualizations
    if not args.skip_visualizations:
        logger.info("Step 2: Creating advanced visualizations...")
        visualizer = ErrorVisualizer(report)
        visualizer.create_visualizations(os.path.join(args.output_dir, "visualizations"))
        logger.info("Advanced visualizations created")

    # Step 3: Analyze error patterns and generate improvement suggestions
    logger.info("Step 3: Analyzing error patterns and generating improvement suggestions...")
    improver = ErrorImprover(report)

    # Analyze error patterns
    error_patterns = improver.analyze_error_patterns()

    # Save error patterns
    patterns_path = os.path.join(args.output_dir, "error_patterns.json")
    with open(patterns_path, 'w') as f:
        json.dump(error_patterns, f, indent=2)

    # Generate and save improvement suggestions
    suggestions_path = improver.save_suggestions(args.output_dir)

    # Print summary
    print("\n===== TEST2 ERROR ANALYSIS SUMMARY =====")
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

    # Print key error patterns
    print("\n===== KEY ERROR PATTERNS =====")

    if error_patterns["phonetic_confusions"]:
        print("\nPhonetic confusions:")
        for i, confusion in enumerate(error_patterns["phonetic_confusions"][:3], 1):
            print(f"{i}. '{confusion['reference']}' -> '{confusion['hypothesis']}' ({confusion['confusion_type']})")

    if error_patterns["substitution_patterns"]:
        print("\nSubstitution patterns:")
        for i, pattern in enumerate(error_patterns["substitution_patterns"][:3], 1):
            subs = [f"'{s['hypothesis']}'" for s in pattern["substitutions"][:2]]
            print(f"{i}. '{pattern['reference']}' -> {', '.join(subs)} ({pattern['pattern_type']})")

    # Print file locations
    print(f"\nDetailed report saved to: {os.path.join(args.output_dir, 'error_analysis_report.json')}")
    print(f"Error patterns saved to: {patterns_path}")
    print(f"Improvement suggestions saved to: {suggestions_path}")

    if not args.skip_visualizations:
        print(f"Visualizations saved to: {os.path.join(args.output_dir, 'visualizations')}")

if __name__ == "__main__":
    main()
