"""
Script to benchmark and compare different decoding methods for speech recognition.
"""

import argparse
import logging
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any

import torch
import numpy as np

from src.preprocessing.audio import batch_preprocess
from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder
from src.decoder.ctc_drafter import CTCDrafter
from src.decoder.cr_ctc import CRCTC
from src.decoder.speculative_decoder import SpeculativeDecoder
from src.utils.file_utils import get_audio_files, save_transcriptions, save_json
from src.utils.evaluation import evaluate_transcriptions, save_evaluation_results, plot_metrics

from config.config import (
    TEST1_DIR, TRANSCRIPTS_DIR, RESULTS_DIR, 
    PRETRAINED_MODEL_NAME, DECODER_TYPE, BEAM_WIDTH
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_decoders.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Decoders for Speech Recognition")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=str(TEST1_DIR),
        help="Directory containing audio files to transcribe"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=str(TRANSCRIPTS_DIR),
        help="Directory to save transcriptions"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=str(RESULTS_DIR),
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--reference_dir", 
        type=str, 
        default=None,
        help="Directory containing reference transcriptions for evaluation"
    )
    
    parser.add_argument(
        "--normalize_audio", 
        action="store_true",
        help="Whether to normalize audio"
    )
    
    parser.add_argument(
        "--remove_silence", 
        action="store_true",
        help="Whether to remove silence from audio"
    )
    
    parser.add_argument(
        "--methods", 
        type=str, 
        nargs="+",
        default=["standard", "ctc_drafter", "cr_ctc", "speculative"],
        help="Decoding methods to benchmark"
    )
    
    parser.add_argument(
        "--small_model", 
        type=str, 
        default="facebook/wav2vec2-base-960h",
        help="Small model for CTC-Drafter"
    )
    
    parser.add_argument(
        "--large_model", 
        type=str, 
        default="facebook/wav2vec2-large-960h-lv60-self",
        help="Large model for verification and CR-CTC"
    )
    
    parser.add_argument(
        "--decoder_type", 
        type=str, 
        default=DECODER_TYPE,
        choices=["greedy", "beam_search"],
        help="Type of CTC decoder to use"
    )
    
    parser.add_argument(
        "--beam_width", 
        type=int, 
        default=BEAM_WIDTH,
        help="Beam width for beam search decoding"
    )
    
    parser.add_argument(
        "--num_perturbations", 
        type=int, 
        default=3,
        help="Number of perturbations for CR-CTC"
    )
    
    return parser.parse_args()

def run_standard_decoding(model, decoder, audio_data_dict):
    """Run standard CTC decoding."""
    logger.info("Running standard CTC decoding")
    start_time = time.time()
    
    results = {}
    transcriptions = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            file_start_time = time.time()
            
            # Get logits from model
            logits = model.get_logits(audio_data, sample_rate)
            
            # Decode using CTC decoder
            transcription = decoder.decode(logits)
            
            file_time = time.time() - file_start_time
            
            results[file_path] = {
                "transcription": transcription,
                "time_ms": file_time * 1000
            }
            
            transcriptions[file_path] = transcription
            
            logger.info(f"Decoded {file_path} in {file_time * 1000:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error decoding {file_path}: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f"Standard decoding completed in {total_time:.2f} seconds")
    
    return results, transcriptions

def run_ctc_drafter(drafter, audio_data_dict):
    """Run CTC-Drafter decoding."""
    logger.info("Running CTC-Drafter decoding")
    start_time = time.time()
    
    results = {}
    transcriptions = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            file_results = drafter.speculative_decode(audio_data, sample_rate)
            results[file_path] = file_results
            transcriptions[file_path] = file_results["verified_transcription"]
            logger.info(f"Decoded {file_path} in {file_results['total_time_ms']:.2f} ms")
        except Exception as e:
            logger.error(f"Error decoding {file_path}: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f"CTC-Drafter decoding completed in {total_time:.2f} seconds")
    
    return results, transcriptions

def run_cr_ctc(cr_ctc, audio_data_dict):
    """Run CR-CTC decoding."""
    logger.info("Running CR-CTC decoding")
    start_time = time.time()
    
    results = {}
    transcriptions = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            file_results = cr_ctc.decode(audio_data, sample_rate)
            results[file_path] = file_results
            transcriptions[file_path] = file_results["final_transcription"]
            logger.info(f"Decoded {file_path} in {file_results['total_time_ms']:.2f} ms")
        except Exception as e:
            logger.error(f"Error decoding {file_path}: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f"CR-CTC decoding completed in {total_time:.2f} seconds")
    
    return results, transcriptions

def run_speculative_decoding(speculative_decoder, audio_data_dict):
    """Run speculative decoding with CR-CTC."""
    logger.info("Running speculative decoding with CR-CTC")
    start_time = time.time()
    
    results = {}
    transcriptions = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            file_results = speculative_decoder.decode(audio_data, sample_rate)
            results[file_path] = file_results
            transcriptions[file_path] = file_results["final_transcription"]
            logger.info(f"Decoded {file_path} in {file_results['total_time_ms']:.2f} ms")
        except Exception as e:
            logger.error(f"Error decoding {file_path}: {str(e)}")
    
    total_time = time.time() - start_time
    logger.info(f"Speculative decoding completed in {total_time:.2f} seconds")
    
    return results, transcriptions

def evaluate_method(transcriptions, references, method_name):
    """Evaluate transcriptions against references."""
    logger.info(f"Evaluating {method_name} transcriptions")
    
    if not references:
        logger.warning("No reference transcriptions for evaluation")
        return None
    
    evaluation_results = evaluate_transcriptions(references, transcriptions)
    
    # Calculate average metrics
    avg_wer = np.mean([r["wer"] for r in evaluation_results.values()])
    avg_cer = np.mean([r["cer"] for r in evaluation_results.values()])
    
    logger.info(f"{method_name} - Average WER: {avg_wer:.4f}, Average CER: {avg_cer:.4f}")
    
    return {
        "method": method_name,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "detailed_results": evaluation_results
    }

def plot_benchmark_results(benchmark_results, output_dir):
    """Plot benchmark results."""
    logger.info("Plotting benchmark results")
    
    # Create DataFrame for plotting
    df = pd.DataFrame(benchmark_results)
    
    # Plot WER comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df["method"], df["avg_wer"], color="blue", alpha=0.7)
    plt.xlabel("Method")
    plt.ylabel("Word Error Rate (WER)")
    plt.title("WER Comparison of Decoding Methods")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wer_comparison.png"))
    
    # Plot CER comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df["method"], df["avg_cer"], color="green", alpha=0.7)
    plt.xlabel("Method")
    plt.ylabel("Character Error Rate (CER)")
    plt.title("CER Comparison of Decoding Methods")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cer_comparison.png"))
    
    # Plot time comparison
    plt.figure(figsize=(12, 6))
    plt.bar(df["method"], df["avg_time_ms"], color="red", alpha=0.7)
    plt.xlabel("Method")
    plt.ylabel("Average Time (ms)")
    plt.title("Time Comparison of Decoding Methods")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    
    logger.info(f"Saved benchmark plots to {output_dir}")

def main():
    """Main function for benchmarking decoders."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Get audio files
    audio_files = get_audio_files(args.input_dir)
    if not audio_files:
        logger.error(f"No audio files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files in {args.input_dir}")
    
    # Preprocess audio
    start_time = time.time()
    logger.info("Preprocessing audio files...")
    audio_data_dict = batch_preprocess(
        audio_files, 
        normalize=args.normalize_audio, 
        remove_silence_flag=args.remove_silence
    )
    logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    # Load reference transcriptions if available
    references = None
    if args.reference_dir:
        from src.utils.file_utils import load_transcriptions
        logger.info(f"Loading reference transcriptions from {args.reference_dir}")
        references = load_transcriptions(args.reference_dir)
    
    # Initialize models and decoders based on methods to benchmark
    models = {}
    decoders = {}
    
    if "standard" in args.methods or "cr_ctc" in args.methods:
        logger.info(f"Initializing large model: {args.large_model}")
        models["large"] = AcousticModel(model_name=args.large_model)
        decoders["large"] = CTCDecoder(
            processor=models["large"].processor,
            decoder_type=args.decoder_type,
            beam_width=args.beam_width
        )
    
    if "ctc_drafter" in args.methods or "speculative" in args.methods:
        logger.info(f"Initializing small model: {args.small_model}")
        models["small"] = AcousticModel(model_name=args.small_model)
        decoders["small"] = CTCDecoder(
            processor=models["small"].processor,
            decoder_type=args.decoder_type,
            beam_width=args.beam_width
        )
        
        if "large" not in models:
            logger.info(f"Initializing large model: {args.large_model}")
            models["large"] = AcousticModel(model_name=args.large_model)
            decoders["large"] = CTCDecoder(
                processor=models["large"].processor,
                decoder_type=args.decoder_type,
                beam_width=args.beam_width
            )
    
    # Initialize specialized decoders
    specialized_decoders = {}
    
    if "ctc_drafter" in args.methods or "speculative" in args.methods:
        logger.info("Initializing CTC-Drafter")
        specialized_decoders["ctc_drafter"] = CTCDrafter(
            drafter_model=models["small"],
            verifier_model=models["large"],
            drafter_decoder=decoders["small"],
            verifier_decoder=decoders["large"]
        )
    
    if "cr_ctc" in args.methods or "speculative" in args.methods:
        logger.info("Initializing CR-CTC")
        specialized_decoders["cr_ctc"] = CRCTC(
            model=models["large"],
            decoder=decoders["large"],
            num_perturbations=args.num_perturbations
        )
    
    if "speculative" in args.methods:
        logger.info("Initializing Speculative Decoder")
        specialized_decoders["speculative"] = SpeculativeDecoder(
            drafter=specialized_decoders["ctc_drafter"],
            verifier=specialized_decoders["cr_ctc"],
            use_cr_ctc_for_verification=True,
            fallback_to_standard_decoding=True
        )
    
    # Run benchmarks
    all_results = {}
    all_transcriptions = {}
    benchmark_results = []
    
    if "standard" in args.methods:
        logger.info("Benchmarking standard decoding")
        results, transcriptions = run_standard_decoding(models["large"], decoders["large"], audio_data_dict)
        all_results["standard"] = results
        all_transcriptions["standard"] = transcriptions
        
        # Calculate average time
        avg_time = np.mean([r["time_ms"] for r in results.values()])
        logger.info(f"Standard decoding - Average time: {avg_time:.2f} ms")
        
        # Evaluate if references are available
        eval_results = None
        if references:
            eval_results = evaluate_method(transcriptions, references, "standard")
        
        benchmark_results.append({
            "method": "standard",
            "avg_time_ms": avg_time,
            "avg_wer": eval_results["avg_wer"] if eval_results else None,
            "avg_cer": eval_results["avg_cer"] if eval_results else None
        })
    
    if "ctc_drafter" in args.methods:
        logger.info("Benchmarking CTC-Drafter")
        results, transcriptions = run_ctc_drafter(specialized_decoders["ctc_drafter"], audio_data_dict)
        all_results["ctc_drafter"] = results
        all_transcriptions["ctc_drafter"] = transcriptions
        
        # Calculate average time
        avg_time = np.mean([r["total_time_ms"] for r in results.values()])
        logger.info(f"CTC-Drafter - Average time: {avg_time:.2f} ms")
        
        # Evaluate if references are available
        eval_results = None
        if references:
            eval_results = evaluate_method(transcriptions, references, "ctc_drafter")
        
        benchmark_results.append({
            "method": "ctc_drafter",
            "avg_time_ms": avg_time,
            "avg_wer": eval_results["avg_wer"] if eval_results else None,
            "avg_cer": eval_results["avg_cer"] if eval_results else None
        })
    
    if "cr_ctc" in args.methods:
        logger.info("Benchmarking CR-CTC")
        results, transcriptions = run_cr_ctc(specialized_decoders["cr_ctc"], audio_data_dict)
        all_results["cr_ctc"] = results
        all_transcriptions["cr_ctc"] = transcriptions
        
        # Calculate average time
        avg_time = np.mean([r["total_time_ms"] for r in results.values()])
        logger.info(f"CR-CTC - Average time: {avg_time:.2f} ms")
        
        # Evaluate if references are available
        eval_results = None
        if references:
            eval_results = evaluate_method(transcriptions, references, "cr_ctc")
        
        benchmark_results.append({
            "method": "cr_ctc",
            "avg_time_ms": avg_time,
            "avg_wer": eval_results["avg_wer"] if eval_results else None,
            "avg_cer": eval_results["avg_cer"] if eval_results else None
        })
    
    if "speculative" in args.methods:
        logger.info("Benchmarking Speculative Decoding")
        results, transcriptions = run_speculative_decoding(specialized_decoders["speculative"], audio_data_dict)
        all_results["speculative"] = results
        all_transcriptions["speculative"] = transcriptions
        
        # Calculate average time
        avg_time = np.mean([r["total_time_ms"] for r in results.values()])
        logger.info(f"Speculative Decoding - Average time: {avg_time:.2f} ms")
        
        # Evaluate if references are available
        eval_results = None
        if references:
            eval_results = evaluate_method(transcriptions, references, "speculative")
        
        benchmark_results.append({
            "method": "speculative",
            "avg_time_ms": avg_time,
            "avg_wer": eval_results["avg_wer"] if eval_results else None,
            "avg_cer": eval_results["avg_cer"] if eval_results else None
        })
    
    # Save benchmark results
    benchmark_path = os.path.join(args.results_dir, "benchmark_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Saved benchmark results to {benchmark_path}")
    
    # Plot benchmark results if references are available
    if references:
        plot_benchmark_results(benchmark_results, args.results_dir)
    
    # Save transcriptions for each method
    for method, transcriptions in all_transcriptions.items():
        method_output_dir = os.path.join(args.output_dir, method)
        os.makedirs(method_output_dir, exist_ok=True)
        
        save_transcriptions(transcriptions, method_output_dir)
        logger.info(f"Saved {method} transcriptions to {method_output_dir}")
    
    logger.info("Benchmark completed successfully")

if __name__ == "__main__":
    main()
