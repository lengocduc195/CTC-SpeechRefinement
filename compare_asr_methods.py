#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare different ASR methods: mock CTC decoding vs. real ASR.

This script compares the results of mock CTC decoding (using ctc_eval.py)
with real ASR (using real_asr.py) on the same audio files.

Usage:
    python compare_asr_methods.py --input_dir data/audio --reference_dir data/transcripts
"""

import os
import argparse
import logging
import time
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from jiwer import wer, cer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("compare_asr.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare ASR Methods")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/audio",
        help="Directory containing audio files to transcribe"
    )
    
    parser.add_argument(
        "--reference_dir", 
        type=str, 
        default="data/transcripts",
        help="Directory containing reference transcriptions"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/comparison",
        help="Directory to save comparison results"
    )
    
    parser.add_argument(
        "--mock_output_dir", 
        type=str, 
        default="results/mock_transcripts",
        help="Directory to save mock transcriptions"
    )
    
    parser.add_argument(
        "--real_output_dir", 
        type=str, 
        default="results/real_transcripts",
        help="Directory to save real transcriptions"
    )
    
    parser.add_argument(
        "--real_model_type", 
        type=str, 
        default="wav2vec2",
        choices=["wav2vec2", "whisper"],
        help="Type of real ASR model to use"
    )
    
    parser.add_argument(
        "--real_model_size", 
        type=str, 
        default="default",
        help="Size of the real ASR model to use"
    )
    
    parser.add_argument(
        "--tokenizers", 
        type=str, 
        default="word",
        help="Comma-separated list of tokenizers to use for mock ASR"
    )
    
    parser.add_argument(
        "--beam_size", 
        type=int, 
        default=10,
        help="Beam size for mock ASR"
    )
    
    return parser.parse_args()

def run_mock_asr(input_dir: str, output_dir: str, tokenizers: str, beam_size: int) -> bool:
    """Run mock ASR using ctc_eval.py."""
    logger.info("Running mock ASR...")
    
    try:
        cmd = [
            "python", "ctc_eval.py",
            "--audio_dir", input_dir,
            "--output_dir", output_dir,
            "--tokenizers", tokenizers,
            "--beam_size", str(beam_size),
            "--generate_transcripts"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running mock ASR: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running mock ASR: {str(e)}")
        return False

def run_real_asr(input_dir: str, output_dir: str, model_type: str, model_size: str) -> bool:
    """Run real ASR using real_asr.py."""
    logger.info("Running real ASR...")
    
    try:
        cmd = [
            "python", "real_asr.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--model_type", model_type,
            "--model_size", model_size
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running real ASR: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running real ASR: {str(e)}")
        return False

def load_transcriptions(directory: str) -> Dict[str, str]:
    """Load transcriptions from files."""
    transcriptions = {}
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return transcriptions
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        transcriptions[base_name] = f.read().strip()
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {str(e)}")
    
    return transcriptions

def evaluate_transcriptions(references: Dict[str, str], transcriptions: Dict[str, str]) -> Dict[str, float]:
    """Evaluate transcriptions against references."""
    results = {
        "wer": [],
        "cer": []
    }
    
    for base_name, reference in references.items():
        if base_name in transcriptions:
            transcription = transcriptions[base_name]
            
            # Calculate WER and CER
            try:
                results["wer"].append(wer(reference, transcription))
                results["cer"].append(cer(reference, transcription))
            except Exception as e:
                logger.error(f"Error calculating metrics for {base_name}: {str(e)}")
    
    # Calculate averages
    if results["wer"]:
        results["avg_wer"] = sum(results["wer"]) / len(results["wer"])
    else:
        results["avg_wer"] = 0.0
    
    if results["cer"]:
        results["avg_cer"] = sum(results["cer"]) / len(results["cer"])
    else:
        results["avg_cer"] = 0.0
    
    return results

def compare_methods(references: Dict[str, str], mock_transcriptions: Dict[str, str], 
                   real_transcriptions: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Compare mock and real ASR methods."""
    # Evaluate mock ASR
    mock_results = evaluate_transcriptions(references, mock_transcriptions)
    
    # Evaluate real ASR
    real_results = evaluate_transcriptions(references, real_transcriptions)
    
    # Compare results
    comparison = {
        "mock": mock_results,
        "real": real_results
    }
    
    return comparison

def plot_comparison(comparison: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Plot comparison results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create bar chart for WER
    plt.figure(figsize=(10, 6))
    methods = list(comparison.keys())
    wer_values = [comparison[method]["avg_wer"] for method in methods]
    
    plt.bar(methods, wer_values)
    plt.xlabel("ASR Method")
    plt.ylabel("Word Error Rate (WER)")
    plt.title("Comparison of ASR Methods - WER")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "wer_comparison.png"))
    plt.close()
    
    # Create bar chart for CER
    plt.figure(figsize=(10, 6))
    cer_values = [comparison[method]["avg_cer"] for method in methods]
    
    plt.bar(methods, cer_values)
    plt.xlabel("ASR Method")
    plt.ylabel("Character Error Rate (CER)")
    plt.title("Comparison of ASR Methods - CER")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "cer_comparison.png"))
    plt.close()

def save_comparison_results(comparison: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Save comparison results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    with open(os.path.join(output_dir, "comparison_results.json"), 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=4)
    
    # Save as CSV
    data = []
    for method, results in comparison.items():
        data.append({
            "Method": method,
            "WER": results["avg_wer"],
            "CER": results["avg_cer"]
        })
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.mock_output_dir, exist_ok=True)
    os.makedirs(args.real_output_dir, exist_ok=True)
    
    # Run mock ASR
    mock_success = run_mock_asr(
        args.input_dir, 
        args.mock_output_dir, 
        args.tokenizers, 
        args.beam_size
    )
    
    # Run real ASR
    real_success = run_real_asr(
        args.input_dir, 
        args.real_output_dir, 
        args.real_model_type, 
        args.real_model_size
    )
    
    if not mock_success or not real_success:
        logger.error("One or both ASR methods failed. Cannot compare results.")
        return
    
    # Load reference transcriptions
    logger.info(f"Loading reference transcriptions from {args.reference_dir}")
    references = load_transcriptions(args.reference_dir)
    
    if not references:
        logger.error("No reference transcriptions found. Cannot evaluate results.")
        return
    
    # Load mock transcriptions
    logger.info(f"Loading mock transcriptions from {args.mock_output_dir}")
    mock_transcriptions = {}
    
    # For each tokenizer, load beam search results
    tokenizers = args.tokenizers.split(',')
    for tokenizer in tokenizers:
        tokenizer_dir = os.path.join(args.mock_output_dir, "generated_transcripts", tokenizer)
        if os.path.exists(tokenizer_dir):
            for root, _, files in os.walk(tokenizer_dir):
                for file in files:
                    if file.endswith('_beam.txt'):
                        file_path = os.path.join(root, file)
                        base_name = os.path.splitext(file)[0].replace('_beam', '')
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                mock_transcriptions[base_name] = f.read().strip()
                        except Exception as e:
                            logger.error(f"Error loading file {file_path}: {str(e)}")
    
    # Load real transcriptions
    logger.info(f"Loading real transcriptions from {args.real_output_dir}")
    real_transcriptions = load_transcriptions(args.real_output_dir)
    
    # Compare methods
    logger.info("Comparing ASR methods...")
    comparison = compare_methods(references, mock_transcriptions, real_transcriptions)
    
    # Save and plot results
    logger.info(f"Saving comparison results to {args.output_dir}")
    save_comparison_results(comparison, args.output_dir)
    plot_comparison(comparison, args.output_dir)
    
    # Print summary
    logger.info("Comparison summary:")
    logger.info(f"Mock ASR - WER: {comparison['mock']['avg_wer']:.4f}, CER: {comparison['mock']['avg_cer']:.4f}")
    logger.info(f"Real ASR - WER: {comparison['real']['avg_wer']:.4f}, CER: {comparison['real']['avg_cer']:.4f}")
    
    logger.info("Comparison completed successfully")

if __name__ == "__main__":
    main()
