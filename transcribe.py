"""
Main script for CTC speech transcription.
"""

import os
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

from src.preprocessing.audio import batch_preprocess
from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder
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
        logging.FileHandler("transcription.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CTC Speech Transcription")
    
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
        "--model_name", 
        type=str, 
        default=PRETRAINED_MODEL_NAME,
        help="Pretrained model name or path"
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
        "--reference_dir", 
        type=str, 
        default=None,
        help="Directory containing reference transcriptions for evaluation"
    )
    
    return parser.parse_args()

def main():
    """Main function for CTC speech transcription."""
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
    
    # Initialize acoustic model
    start_time = time.time()
    logger.info(f"Initializing acoustic model: {args.model_name}")
    model = AcousticModel(model_name=args.model_name)
    logger.info(f"Model initialization completed in {time.time() - start_time:.2f} seconds")
    
    # Initialize CTC decoder
    logger.info(f"Initializing CTC decoder: {args.decoder_type}")
    decoder = CTCDecoder(
        processor=model.processor,
        decoder_type=args.decoder_type,
        beam_width=args.beam_width
    )
    
    # Process audio and get logits
    start_time = time.time()
    logger.info("Processing audio with the model...")
    logits_dict = model.batch_process(audio_data_dict)
    logger.info(f"Model processing completed in {time.time() - start_time:.2f} seconds")
    
    # Decode logits to get transcriptions
    start_time = time.time()
    logger.info("Decoding logits to get transcriptions...")
    transcriptions = decoder.batch_decode(logits_dict)
    logger.info(f"Decoding completed in {time.time() - start_time:.2f} seconds")
    
    # Save transcriptions
    logger.info(f"Saving transcriptions to {args.output_dir}")
    output_files = save_transcriptions(transcriptions, args.output_dir)
    
    # Save results summary
    results_summary = {
        "model_name": args.model_name,
        "decoder_type": args.decoder_type,
        "beam_width": args.beam_width,
        "normalize_audio": args.normalize_audio,
        "remove_silence": args.remove_silence,
        "num_files": len(audio_files),
        "transcriptions": {os.path.basename(k): v for k, v in transcriptions.items()}
    }
    
    summary_path = os.path.join(args.results_dir, "transcription_summary.json")
    save_json(results_summary, summary_path)
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
    
    logger.info("Transcription process completed successfully")

if __name__ == "__main__":
    main()
