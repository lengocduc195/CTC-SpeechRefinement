"""
Speculative Decoder module combining CTC-Drafter and CR-CTC for efficient and accurate speech recognition.

This module implements a speculative decoding approach that uses CTC-Drafter for fast initial
transcription and CR-CTC for consistency-regularized verification and refinement.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import os
from pathlib import Path
import time

from ctc_speech_refinement.core.models.acoustic_model import AcousticModel
from ctc_speech_refinement.core.decoder.ctc_decoder import CTCDecoder
from ctc_speech_refinement.core.decoder.ctc_drafter import CTCDrafter
from ctc_speech_refinement.core.decoder.cr_ctc import CRCTC

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeculativeDecoder:
    """
    Speculative Decoder combining CTC-Drafter and CR-CTC for efficient and accurate speech recognition.
    
    This decoder uses a two-stage approach:
    1. CTC-Drafter for fast initial transcription
    2. CR-CTC for consistency-regularized verification and refinement
    """
    
    def __init__(self, 
                drafter: CTCDrafter,
                verifier: CRCTC,
                use_cr_ctc_for_verification: bool = True,
                fallback_to_standard_decoding: bool = True,
                acceptance_threshold: float = 0.5):
        """
        Initialize the Speculative Decoder.
        
        Args:
            drafter: CTC-Drafter for initial transcription.
            verifier: CR-CTC for verification and refinement.
            use_cr_ctc_for_verification: Whether to use CR-CTC for verification.
                If False, only the base verifier model is used.
            fallback_to_standard_decoding: Whether to fall back to standard decoding
                if the acceptance rate is below the threshold.
            acceptance_threshold: Threshold for acceptance rate to determine
                whether to use the draft or fall back to standard decoding.
        """
        self.drafter = drafter
        self.verifier = verifier
        self.use_cr_ctc_for_verification = use_cr_ctc_for_verification
        self.fallback_to_standard_decoding = fallback_to_standard_decoding
        self.acceptance_threshold = acceptance_threshold
        
        logger.info("Initialized Speculative Decoder")
        logger.info(f"Use CR-CTC for verification: {use_cr_ctc_for_verification}")
        logger.info(f"Fallback to standard decoding: {fallback_to_standard_decoding}")
        logger.info(f"Acceptance threshold: {acceptance_threshold}")
    
    def decode(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Perform speculative decoding with CR-CTC verification on audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Dictionary containing transcription results and timing information.
        """
        logger.info("Performing speculative decoding with CR-CTC verification")
        total_start_time = time.time()
        
        # Step 1: Generate draft transcription using CTC-Drafter
        draft_start_time = time.time()
        draft_results = self.drafter.speculative_decode(audio_data, sample_rate)
        draft_time = time.time() - draft_start_time
        
        draft_transcription = draft_results["verified_transcription"]
        draft_acceptance_rate = draft_results["acceptance_rate"]
        
        # Step 2: Decide whether to use the draft or fall back to standard decoding
        use_draft = True
        if self.fallback_to_standard_decoding and draft_acceptance_rate < self.acceptance_threshold:
            logger.info(f"Draft acceptance rate {draft_acceptance_rate:.2f} below threshold {self.acceptance_threshold}")
            logger.info("Falling back to standard decoding")
            use_draft = False
        
        # Step 3: Verify and refine using CR-CTC if needed
        verification_results = None
        verification_time = 0
        
        if self.use_cr_ctc_for_verification:
            verification_start_time = time.time()
            verification_results = self.verifier.decode(audio_data, sample_rate)
            verification_time = time.time() - verification_start_time
            
            final_transcription = verification_results["final_transcription"]
        else:
            # If not using CR-CTC, the final transcription is the draft transcription
            final_transcription = draft_transcription
        
        total_time = time.time() - total_start_time
        
        # Prepare results
        results = {
            "draft_transcription": draft_results["draft_transcription"],
            "verified_draft_transcription": draft_results["verified_transcription"],
            "final_transcription": final_transcription,
            "draft_time_ms": draft_results["total_time_ms"],
            "verification_time_ms": verification_time * 1000 if verification_results else 0,
            "total_time_ms": total_time * 1000,
            "draft_acceptance_rate": draft_acceptance_rate,
            "used_draft": use_draft,
            "used_cr_ctc": self.use_cr_ctc_for_verification
        }
        
        # Add CR-CTC specific results if available
        if verification_results:
            results["cr_ctc_results"] = {
                "perturbation_time_ms": verification_results["perturbation_time_ms"],
                "decoding_time_ms": verification_results["decoding_time_ms"],
                "voting_time_ms": verification_results["voting_time_ms"],
                "num_perturbations": verification_results["num_perturbations"]
            }
        
        logger.info(f"Speculative decoding with CR-CTC completed in {total_time * 1000:.2f} ms")
        logger.info(f"Final transcription: {final_transcription}")
        
        return results
    
    def batch_decode(self, audio_data_dict: Dict[str, Tuple[np.ndarray, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Perform speculative decoding with CR-CTC verification on a batch of audio files.
        
        Args:
            audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
            
        Returns:
            Dictionary mapping file paths to decoding results.
        """
        logger.info(f"Batch speculative decoding with CR-CTC for {len(audio_data_dict)} files")
        results = {}
        
        for file_path, (audio_data, sample_rate) in audio_data_dict.items():
            try:
                file_results = self.decode(audio_data, sample_rate)
                results[file_path] = file_results
                logger.info(f"Completed speculative decoding with CR-CTC for {file_path}")
            except Exception as e:
                logger.error(f"Error decoding {file_path}: {str(e)}")
        
        # Calculate average statistics
        if results:
            avg_draft_time = np.mean([r["draft_time_ms"] for r in results.values()])
            avg_verification_time = np.mean([r["verification_time_ms"] for r in results.values()])
            avg_total_time = np.mean([r["total_time_ms"] for r in results.values()])
            avg_draft_acceptance_rate = np.mean([r["draft_acceptance_rate"] for r in results.values()])
            
            logger.info(f"Average draft time: {avg_draft_time:.2f} ms")
            logger.info(f"Average verification time: {avg_verification_time:.2f} ms")
            logger.info(f"Average total time: {avg_total_time:.2f} ms")
            logger.info(f"Average draft acceptance rate: {avg_draft_acceptance_rate:.2f}")
            
            # Count how many used draft vs. standard decoding
            draft_count = sum(1 for r in results.values() if r["used_draft"])
            standard_count = len(results) - draft_count
            logger.info(f"Used draft: {draft_count}, Used standard: {standard_count}")
        
        return results
