"""
Script to run speculative decoding with CTC-Drafter and CR-CTC for speech recognition.
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
    
    # Verifier model settings
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
    
    # Speculative decoding settings
    parser.add_argument(
        "--max_draft_length", 
        type=int, 
        default=30,
        help="Maximum length of draft sequences"
    )
    
    parser.add_argument(
        "--draft_timeout_ms", 
        type=int, 
        default=50,
        help="Maximum time in milliseconds to spend on drafting"
    )
    
    parser.add_argument(
        "--acceptance_threshold", 
        type=float, 
        default=0.5,
        help="Threshold for acceptance rate to determine whether to use the draft"
    )
    
    # CR-CTC settings
    parser.add_argument(
        "--use_cr_ctc", 
        action="store_true",
        help="Whether to use CR-CTC for verification"
    )
    
    parser.add_argument(
        "--num_perturbations", 
        type=int, 
        default=3,
        help="Number of perturbations to generate for CR-CTC"
    )
    
    parser.add_argument(
        "--fallback_to_standard", 
        action="store_true",
        help="Whether to fall back to standard decoding if acceptance rate is below threshold"
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
    """Main function for speculative decoding with CTC-Drafter and CR-CTC."""
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
    
    # Initialize drafter model
    start_time = time.time()
    logger.info(f"Initializing drafter model: {args.drafter_model}")
    drafter_model = AcousticModel(model_name=args.drafter_model)
    logger.info(f"Drafter model initialization completed in {time.time() - start_time:.2f} seconds")
    
    # Initialize verifier model
    start_time = time.time()
    logger.info(f"Initializing verifier model: {args.verifier_model}")
    verifier_model = AcousticModel(model_name=args.verifier_model)
    logger.info(f"Verifier model initialization completed in {time.time() - start_time:.2f} seconds")
    
    # Initialize CTC decoders
    logger.info(f"Initializing CTC decoders: {args.decoder_type}")
    drafter_decoder = CTCDecoder(
        processor=drafter_model.processor,
        decoder_type=args.decoder_type,
        beam_width=args.beam_width
    )
    
    verifier_decoder = CTCDecoder(
        processor=verifier_model.processor,
        decoder_type=args.decoder_type,
        beam_width=args.beam_width
    )
    
    # Initialize CTC-Drafter
    logger.info("Initializing CTC-Drafter")
    drafter = CTCDrafter(
        drafter_model=drafter_model,
        verifier_model=verifier_model,
        drafter_decoder=drafter_decoder,
        verifier_decoder=verifier_decoder,
        max_draft_length=args.max_draft_length,
        draft_timeout_ms=args.draft_timeout_ms
    )
    
    # Initialize CR-CTC
    logger.info("Initializing CR-CTC")
    cr_ctc = CRCTC(
        model=verifier_model,
        decoder=verifier_decoder,
        num_perturbations=args.num_perturbations
    )
    
    # Initialize Speculative Decoder
    logger.info("Initializing Speculative Decoder")
    speculative_decoder = SpeculativeDecoder(
        drafter=drafter,
        verifier=cr_ctc,
        use_cr_ctc_for_verification=args.use_cr_ctc,
        fallback_to_standard_decoding=args.fallback_to_standard,
        acceptance_threshold=args.acceptance_threshold
    )
    
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
    
    # Convert results to JSON-serializable format
    json_results = {}
    for file_path, result in decoding_results.items():
        # Convert numpy arrays and other non-serializable types
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif isinstance(value, np.float32) or isinstance(value, np.float64):
                json_result[key] = float(value)
            elif isinstance(value, np.int32) or isinstance(value, np.int64):
                json_result[key] = int(value)
            else:
                json_result[key] = value
        
        json_results[file_path] = json_result
    
    with open(detailed_results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Saved detailed results to {detailed_results_path}")
    
    # Save results summary
    results_summary = {
        "drafter_model": args.drafter_model,
        "verifier_model": args.verifier_model,
        "decoder_type": args.decoder_type,
        "beam_width": args.beam_width,
        "use_cr_ctc": args.use_cr_ctc,
        "num_perturbations": args.num_perturbations,
        "fallback_to_standard": args.fallback_to_standard,
        "acceptance_threshold": args.acceptance_threshold,
        "normalize_audio": args.normalize_audio,
        "remove_silence": args.remove_silence,
        "num_files": len(audio_files),
        "avg_draft_time_ms": np.mean([r["draft_time_ms"] for r in decoding_results.values()]),
        "avg_verification_time_ms": np.mean([r["verification_time_ms"] for r in decoding_results.values()]),
        "avg_total_time_ms": np.mean([r["total_time_ms"] for r in decoding_results.values()]),
        "avg_draft_acceptance_rate": np.mean([r["draft_acceptance_rate"] for r in decoding_results.values()]),
        "draft_usage_count": sum(1 for r in decoding_results.values() if r["used_draft"]),
        "standard_usage_count": sum(1 for r in decoding_results.values() if not r["used_draft"])
    }
    
    summary_path = os.path.join(args.results_dir, "speculative_decoding_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Saved results summary to {summary_path}")
    
    # Evaluate if reference transcriptions are available
    if args.reference_dir:
        from src.utils.file_utils import load_transcriptions
        
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
    
    logger.info("Speculative decoding process completed successfully")

if __name__ == "__main__":
    main()
