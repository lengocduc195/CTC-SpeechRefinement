"""
CTC-Drafter module for speculative decoding in speech recognition.

This module implements the CTC-Drafter approach for speculative decoding,
which uses a smaller, faster model to propose candidate transcriptions
that are then verified by a larger, more accurate model.
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTCDrafter:
    """
    CTC-Drafter for speculative decoding in speech recognition.
    
    The CTC-Drafter uses a smaller, faster model to propose candidate transcriptions
    that are then verified by a larger, more accurate model.
    """
    
    def __init__(self, 
                drafter_model: AcousticModel,
                verifier_model: AcousticModel,
                drafter_decoder: CTCDecoder,
                verifier_decoder: CTCDecoder,
                max_draft_length: int = 30,
                draft_timeout_ms: int = 50):
        """
        Initialize the CTC-Drafter.
        
        Args:
            drafter_model: Smaller, faster acoustic model for drafting.
            verifier_model: Larger, more accurate acoustic model for verification.
            drafter_decoder: CTC decoder for the drafter model.
            verifier_decoder: CTC decoder for the verifier model.
            max_draft_length: Maximum length of draft sequences.
            draft_timeout_ms: Maximum time in milliseconds to spend on drafting.
        """
        self.drafter_model = drafter_model
        self.verifier_model = verifier_model
        self.drafter_decoder = drafter_decoder
        self.verifier_decoder = verifier_decoder
        self.max_draft_length = max_draft_length
        self.draft_timeout_ms = draft_timeout_ms
        
        logger.info("Initialized CTC-Drafter")
    
    def draft(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Generate a draft transcription using the drafter model.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Draft transcription.
        """
        logger.info("Generating draft transcription")
        start_time = time.time()
        
        # Get logits from drafter model
        drafter_logits = self.drafter_model.get_logits(audio_data, sample_rate)
        
        # Decode using drafter decoder
        draft_transcription = self.drafter_decoder.decode(drafter_logits)
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Draft generation took {elapsed_ms:.2f} ms")
        logger.info(f"Draft transcription: {draft_transcription}")
        
        return draft_transcription
    
    def verify(self, audio_data: np.ndarray, sample_rate: int, draft_transcription: str) -> str:
        """
        Verify and correct a draft transcription using the verifier model.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            draft_transcription: Draft transcription to verify.
            
        Returns:
            Verified and corrected transcription.
        """
        logger.info("Verifying draft transcription")
        start_time = time.time()
        
        # Get logits from verifier model
        verifier_logits = self.verifier_model.get_logits(audio_data, sample_rate)
        
        # Decode using verifier decoder
        verified_transcription = self.verifier_decoder.decode(verifier_logits)
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Verification took {elapsed_ms:.2f} ms")
        logger.info(f"Verified transcription: {verified_transcription}")
        
        return verified_transcription
    
    def speculative_decode(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Perform speculative decoding on audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Dictionary containing transcription results and timing information.
        """
        logger.info("Performing speculative decoding")
        total_start_time = time.time()
        
        # Step 1: Generate draft transcription
        draft_start_time = time.time()
        draft_transcription = self.draft(audio_data, sample_rate)
        draft_time = time.time() - draft_start_time
        
        # Step 2: Verify and correct the draft
        verify_start_time = time.time()
        verified_transcription = self.verify(audio_data, sample_rate, draft_transcription)
        verify_time = time.time() - verify_start_time
        
        # Calculate acceptance rate (how much of the draft was accepted)
        # This is a simple character-level Levenshtein distance-based metric
        from Levenshtein import distance
        draft_len = len(draft_transcription)
        if draft_len > 0:
            edit_distance = distance(draft_transcription, verified_transcription)
            acceptance_rate = max(0, 1 - edit_distance / draft_len)
        else:
            acceptance_rate = 0.0
        
        total_time = time.time() - total_start_time
        
        # Prepare results
        results = {
            "draft_transcription": draft_transcription,
            "verified_transcription": verified_transcription,
            "draft_time_ms": draft_time * 1000,
            "verify_time_ms": verify_time * 1000,
            "total_time_ms": total_time * 1000,
            "acceptance_rate": acceptance_rate
        }
        
        logger.info(f"Speculative decoding completed in {total_time * 1000:.2f} ms")
        logger.info(f"Acceptance rate: {acceptance_rate:.2f}")
        
        return results
    
    def batch_speculative_decode(self, audio_data_dict: Dict[str, Tuple[np.ndarray, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Perform speculative decoding on a batch of audio files.
        
        Args:
            audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
            
        Returns:
            Dictionary mapping file paths to decoding results.
        """
        logger.info(f"Batch speculative decoding for {len(audio_data_dict)} files")
        results = {}
        
        for file_path, (audio_data, sample_rate) in audio_data_dict.items():
            try:
                file_results = self.speculative_decode(audio_data, sample_rate)
                results[file_path] = file_results
                logger.info(f"Completed speculative decoding for {file_path}")
            except Exception as e:
                logger.error(f"Error decoding {file_path}: {str(e)}")
        
        # Calculate average statistics
        if results:
            avg_draft_time = np.mean([r["draft_time_ms"] for r in results.values()])
            avg_verify_time = np.mean([r["verify_time_ms"] for r in results.values()])
            avg_total_time = np.mean([r["total_time_ms"] for r in results.values()])
            avg_acceptance_rate = np.mean([r["acceptance_rate"] for r in results.values()])
            
            logger.info(f"Average draft time: {avg_draft_time:.2f} ms")
            logger.info(f"Average verify time: {avg_verify_time:.2f} ms")
            logger.info(f"Average total time: {avg_total_time:.2f} ms")
            logger.info(f"Average acceptance rate: {avg_acceptance_rate:.2f}")
        
        return results
