"""
Module for running speculative decoding with CTC-Drafter and CR-CTC.
"""

import argparse
import logging
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any

import torch
import numpy as np

from ctc_speech_refinement.core.preprocessing.audio import batch_preprocess
from ctc_speech_refinement.core.models.acoustic_model import AcousticModel
from ctc_speech_refinement.core.decoder.ctc_decoder import CTCDecoder
from ctc_speech_refinement.core.decoder.ctc_drafter import CTCDrafter
from ctc_speech_refinement.core.decoder.cr_ctc import CRCTC
from ctc_speech_refinement.core.decoder.speculative_decoder import SpeculativeDecoder
from ctc_speech_refinement.core.utils.file_utils import get_audio_files, save_transcriptions, save_json
from ctc_speech_refinement.core.utils.evaluation import evaluate_transcriptions, save_evaluation_results, plot_metrics
from ctc_speech_refinement.config.config import (
    TEST1_DIR, TRANSCRIPTS_DIR, RESULTS_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speculative_decoding.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Speculative Decoding with CTC-Drafter and CR-CTC")
    
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
    
    # Drafter model settings
    parser.add_argument(
        "--drafter_model", 
        type=str, 
        default="facebook/wav2vec2-base-960h",
        help="Pretrained model name or path for the drafter"
    )
    
    parser.add_argument(
        "--verifier_model", 
        type=str, 
        default="facebook/wav2vec2-large-960h-lv60-self",
        help="Pretrained model name or path for the verifier"
    )
    
    # Decoder settings
    parser.add_argument(
        "--decoder_type", 
        type=str, 
        default="greedy",
        choices=["greedy", "beam_search"],
        help="Type of CTC decoder to use"
    )
    
    parser.add_argument(
        "--beam_width", 
        type=int, 
        default=100,
        help="Beam width for beam search decoding"
    )
    
    # Speculative decoding settings
    parser.add_argument(
        "--use_cr_ctc", 
        action="store_true",
        help="Whether to use CR-CTC"
    )
    
    parser.add_argument(
        "--consistency_alpha", 
        type=float, 
        default=0.5,
        help="Alpha parameter for CR-CTC"
    )
    
    parser.add_argument(
        "--max_draft_length", 
        type=int, 
        default=100,
        help="Maximum length of draft transcription"
    )
    
    # Audio preprocessing settings
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
    
    # Evaluation settings
    parser.add_argument(
        "--reference_dir", 
        type=str, 
        default=None,
        help="Directory containing reference transcriptions for evaluation"
    )
    
    return parser.parse_args()

def main():
    """Main function for speculative decoding."""
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
    
    # Initialize speculative decoder
    start_time = time.time()
    logger.info("Initializing speculative decoder...")
    speculative_decoder = SpeculativeDecoder(
        drafter_model_name=args.drafter_model,
        verifier_model_name=args.verifier_model,
        decoder_type=args.decoder_type,
        beam_width=args.beam_width,
        use_cr_ctc=args.use_cr_ctc,
        consistency_alpha=args.consistency_alpha,
        max_draft_length=args.max_draft_length
    )
    logger.info(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    
    # Perform speculative decoding
    start_time = time.time()
    logger.info("Performing speculative decoding...")
    decoding_results = speculative_decoder.batch_decode(audio_data_dict)
    logger.info(f"Decoding completed in {time.time() - start_time:.2f} seconds")
    
    # Extract transcriptions
    transcriptions = {}
    for file_path, result in decoding_results.items():
        transcriptions[file_path] = result["final_transcription"]
    
    # Save transcriptions
    logger.info(f"Saving transcriptions to {args.output_dir}")
    output_files = save_transcriptions(transcriptions, args.output_dir)
    
    # Save detailed results
    detailed_results_path = os.path.join(args.results_dir, "speculative_decoding_results.json")
    save_json(decoding_results, detailed_results_path)
    logger.info(f"Saved detailed results to {detailed_results_path}")
    
    # Print summary
    print("\n===== SPECULATIVE DECODING SUMMARY =====")
    print(f"Number of files processed: {len(decoding_results)}")
    
    # Calculate average metrics
    avg_draft_time = np.mean([result["draft_time_ms"] for result in decoding_results.values()])
    avg_verify_time = np.mean([result["verify_time_ms"] for result in decoding_results.values()])
    avg_total_time = np.mean([result["total_time_ms"] for result in decoding_results.values()])
    avg_acceptance_rate = np.mean([result["acceptance_rate"] for result in decoding_results.values()])
    
    print(f"Average draft time: {avg_draft_time:.2f} ms")
    print(f"Average verify time: {avg_verify_time:.2f} ms")
    print(f"Average total time: {avg_total_time:.2f} ms")
    print(f"Average acceptance rate: {avg_acceptance_rate:.2f}")
    
    # Evaluate if reference transcriptions are available
    if args.reference_dir:
        from ctc_speech_refinement.core.utils.file_utils import load_transcriptions
        
        logger.info(f"Loading reference transcriptions from {args.reference_dir}")
        references = load_transcriptions(args.reference_dir)
        
        if references:
            logger.info("Evaluating transcriptions against references")
            evaluation_results = evaluate_transcriptions(references, transcriptions)
            
            # Save evaluation results
            eval_path = save_evaluation_results(evaluation_results, args.results_dir)
            logger.info(f"Saved evaluation results to {eval_path}")
            
            # Plot metrics
            plot_path = plot_metrics(evaluation_results, args.results_dir)
            logger.info(f"Saved metrics plot to {plot_path}")
        else:
            logger.warning("No reference transcriptions found for evaluation")
    
    logger.info("Speculative decoding completed successfully")

if __name__ == "__main__":
    main()
