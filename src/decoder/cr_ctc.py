"""
Consistency-Regularized CTC (CR-CTC) module for improved speech recognition.

This module implements the CR-CTC approach, which enhances CTC decoding
by enforcing consistency between different perturbations of the input.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import os
from pathlib import Path
import time

from src.models.acoustic_model import AcousticModel
from src.decoder.ctc_decoder import CTCDecoder
from src.eda.preprocessing import time_stretch, pitch_shift, add_noise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CRCTC:
    """
    Consistency-Regularized CTC (CR-CTC) for improved speech recognition.
    
    CR-CTC enhances CTC decoding by enforcing consistency between different
    perturbations of the input, leading to more robust transcriptions.
    """
    
    def __init__(self, 
                model: AcousticModel,
                decoder: CTCDecoder,
                num_perturbations: int = 3,
                time_stretch_factors: List[float] = [0.9, 1.0, 1.1],
                pitch_shift_steps: List[float] = [-1.0, 0.0, 1.0],
                noise_levels: List[float] = [0.0, 0.005, 0.01],
                consistency_threshold: float = 0.7):
        """
        Initialize the CR-CTC decoder.
        
        Args:
            model: Acoustic model for transcription.
            decoder: CTC decoder for the model.
            num_perturbations: Number of perturbations to generate.
            time_stretch_factors: List of time stretch factors to apply.
            pitch_shift_steps: List of pitch shift steps to apply (in semitones).
            noise_levels: List of noise levels to add.
            consistency_threshold: Threshold for consistency voting.
        """
        self.model = model
        self.decoder = decoder
        self.num_perturbations = num_perturbations
        self.time_stretch_factors = time_stretch_factors
        self.pitch_shift_steps = pitch_shift_steps
        self.noise_levels = noise_levels
        self.consistency_threshold = consistency_threshold
        
        logger.info("Initialized CR-CTC decoder")
    
    def generate_perturbations(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[np.ndarray, str]]:
        """
        Generate perturbed versions of the input audio.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            List of tuples containing (perturbed_audio, perturbation_description).
        """
        logger.info(f"Generating {self.num_perturbations} perturbations")
        perturbations = []
        
        # Always include the original audio
        perturbations.append((audio_data.copy(), "original"))
        
        # Generate time-stretched versions
        for factor in self.time_stretch_factors:
            if factor != 1.0 and len(perturbations) < self.num_perturbations:
                try:
                    stretched_audio = time_stretch(audio_data.copy(), factor)
                    perturbations.append((stretched_audio, f"time_stretch_{factor}"))
                except Exception as e:
                    logger.warning(f"Error generating time stretch perturbation with factor {factor}: {str(e)}")
        
        # Generate pitch-shifted versions
        for steps in self.pitch_shift_steps:
            if steps != 0.0 and len(perturbations) < self.num_perturbations:
                try:
                    shifted_audio = pitch_shift(audio_data.copy(), sample_rate, steps)
                    perturbations.append((shifted_audio, f"pitch_shift_{steps}"))
                except Exception as e:
                    logger.warning(f"Error generating pitch shift perturbation with steps {steps}: {str(e)}")
        
        # Generate noisy versions
        for level in self.noise_levels:
            if level > 0.0 and len(perturbations) < self.num_perturbations:
                try:
                    noisy_audio = add_noise(audio_data.copy(), level)
                    perturbations.append((noisy_audio, f"noise_{level}"))
                except Exception as e:
                    logger.warning(f"Error generating noise perturbation with level {level}: {str(e)}")
        
        # Limit to the requested number of perturbations
        perturbations = perturbations[:self.num_perturbations]
        
        logger.info(f"Generated {len(perturbations)} perturbations")
        return perturbations
    
    def decode_perturbations(self, perturbations: List[Tuple[np.ndarray, str]], sample_rate: int) -> List[Tuple[str, str]]:
        """
        Decode each perturbed audio to get transcriptions.
        
        Args:
            perturbations: List of tuples containing (perturbed_audio, perturbation_description).
            sample_rate: Sample rate of the audio.
            
        Returns:
            List of tuples containing (transcription, perturbation_description).
        """
        logger.info("Decoding perturbations")
        transcriptions = []
        
        for perturbed_audio, description in perturbations:
            try:
                # Get logits from model
                logits = self.model.get_logits(perturbed_audio, sample_rate)
                
                # Decode using CTC decoder
                transcription = self.decoder.decode(logits)
                
                transcriptions.append((transcription, description))
                logger.info(f"Decoded {description}: {transcription}")
            except Exception as e:
                logger.error(f"Error decoding {description}: {str(e)}")
        
        return transcriptions
    
    def apply_consistency_voting(self, transcriptions: List[Tuple[str, str]]) -> str:
        """
        Apply consistency voting to select the most consistent transcription.
        
        Args:
            transcriptions: List of tuples containing (transcription, perturbation_description).
            
        Returns:
            Most consistent transcription.
        """
        logger.info("Applying consistency voting")
        
        if not transcriptions:
            logger.warning("No transcriptions to vote on")
            return ""
        
        # If only one transcription, return it
        if len(transcriptions) == 1:
            return transcriptions[0][0]
        
        # Count word occurrences across all transcriptions
        all_words = []
        for transcription, _ in transcriptions:
            words = transcription.lower().split()
            all_words.extend(words)
        
        # Count occurrences of each word
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Calculate consistency score for each transcription
        scores = []
        for transcription, description in transcriptions:
            words = transcription.lower().split()
            if not words:
                scores.append((0.0, transcription, description))
                continue
            
            # Score is the average frequency of words in the transcription
            total_freq = sum(word_counts[word] for word in words)
            avg_freq = total_freq / len(words)
            normalized_score = avg_freq / len(transcriptions)
            
            scores.append((normalized_score, transcription, description))
        
        # Sort by score in descending order
        scores.sort(reverse=True)
        
        # Log scores
        for score, transcription, description in scores:
            logger.info(f"Score {score:.4f} for {description}: {transcription}")
        
        # Return the transcription with the highest score
        best_score, best_transcription, best_description = scores[0]
        logger.info(f"Selected transcription from {best_description} with score {best_score:.4f}")
        
        return best_transcription
    
    def decode(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Perform CR-CTC decoding on audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Dictionary containing transcription results and timing information.
        """
        logger.info("Performing CR-CTC decoding")
        total_start_time = time.time()
        
        # Step 1: Generate perturbations
        perturbation_start_time = time.time()
        perturbations = self.generate_perturbations(audio_data, sample_rate)
        perturbation_time = time.time() - perturbation_start_time
        
        # Step 2: Decode each perturbation
        decoding_start_time = time.time()
        transcriptions = self.decode_perturbations(perturbations, sample_rate)
        decoding_time = time.time() - decoding_start_time
        
        # Step 3: Apply consistency voting
        voting_start_time = time.time()
        final_transcription = self.apply_consistency_voting(transcriptions)
        voting_time = time.time() - voting_start_time
        
        total_time = time.time() - total_start_time
        
        # Prepare results
        results = {
            "final_transcription": final_transcription,
            "perturbation_time_ms": perturbation_time * 1000,
            "decoding_time_ms": decoding_time * 1000,
            "voting_time_ms": voting_time * 1000,
            "total_time_ms": total_time * 1000,
            "num_perturbations": len(perturbations),
            "perturbation_transcriptions": [
                {"transcription": t, "description": d} for t, d in transcriptions
            ]
        }
        
        logger.info(f"CR-CTC decoding completed in {total_time * 1000:.2f} ms")
        
        return results
    
    def batch_decode(self, audio_data_dict: Dict[str, Tuple[np.ndarray, int]]) -> Dict[str, Dict[str, Any]]:
        """
        Perform CR-CTC decoding on a batch of audio files.
        
        Args:
            audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
            
        Returns:
            Dictionary mapping file paths to decoding results.
        """
        logger.info(f"Batch CR-CTC decoding for {len(audio_data_dict)} files")
        results = {}
        
        for file_path, (audio_data, sample_rate) in audio_data_dict.items():
            try:
                file_results = self.decode(audio_data, sample_rate)
                results[file_path] = file_results
                logger.info(f"Completed CR-CTC decoding for {file_path}")
            except Exception as e:
                logger.error(f"Error decoding {file_path}: {str(e)}")
        
        # Calculate average statistics
        if results:
            avg_perturbation_time = np.mean([r["perturbation_time_ms"] for r in results.values()])
            avg_decoding_time = np.mean([r["decoding_time_ms"] for r in results.values()])
            avg_voting_time = np.mean([r["voting_time_ms"] for r in results.values()])
            avg_total_time = np.mean([r["total_time_ms"] for r in results.values()])
            
            logger.info(f"Average perturbation time: {avg_perturbation_time:.2f} ms")
            logger.info(f"Average decoding time: {avg_decoding_time:.2f} ms")
            logger.info(f"Average voting time: {avg_voting_time:.2f} ms")
            logger.info(f"Average total time: {avg_total_time:.2f} ms")
        
        return results
