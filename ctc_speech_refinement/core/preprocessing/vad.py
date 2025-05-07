"""
Voice Activity Detection (VAD) module for audio preprocessing.

This module provides functions for detecting speech segments in audio data
using various VAD techniques.
"""

import numpy as np
import librosa
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import torch
import os
from pathlib import Path
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def energy_vad(audio_data: np.ndarray, sample_rate: int, 
              frame_length: int = 1024, hop_length: int = 512,
              energy_threshold: float = 0.01, min_speech_duration: float = 0.1) -> List[Tuple[float, float]]:
    """
    Detect speech segments using energy-based Voice Activity Detection.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        frame_length: Length of each frame in samples.
        hop_length: Number of samples between frames.
        energy_threshold: Energy threshold above which audio is considered speech.
        min_speech_duration: Minimum duration of speech in seconds.
        
    Returns:
        List of (start_time, end_time) tuples for speech regions.
    """
    logger.info("Performing energy-based VAD")
    
    # Calculate energy
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find frames above threshold
    speech_frames = np.where(energy > energy_threshold)[0]
    
    if len(speech_frames) == 0:
        logger.warning("No speech detected")
        return []
    
    # Group consecutive frames
    speech_regions = []
    region_start = speech_frames[0]
    
    for i in range(1, len(speech_frames)):
        if speech_frames[i] > speech_frames[i-1] + 1:
            # End of a region
            region_end = speech_frames[i-1]
            
            # Convert to time
            start_time = librosa.frames_to_time(region_start, sr=sample_rate, hop_length=hop_length)
            end_time = librosa.frames_to_time(region_end, sr=sample_rate, hop_length=hop_length)
            
            # Check if duration is long enough
            if end_time - start_time >= min_speech_duration:
                speech_regions.append((start_time, end_time))
            
            # Start of a new region
            region_start = speech_frames[i]
    
    # Add the last region
    if len(speech_frames) > 0:
        region_end = speech_frames[-1]
        start_time = librosa.frames_to_time(region_start, sr=sample_rate, hop_length=hop_length)
        end_time = librosa.frames_to_time(region_end, sr=sample_rate, hop_length=hop_length)
        
        if end_time - start_time >= min_speech_duration:
            speech_regions.append((start_time, end_time))
    
    logger.info(f"Detected {len(speech_regions)} speech regions")
    return speech_regions

def zcr_vad(audio_data: np.ndarray, sample_rate: int,
           frame_length: int = 1024, hop_length: int = 512,
           zcr_threshold: float = 0.2, energy_threshold: float = 0.01,
           min_speech_duration: float = 0.1) -> List[Tuple[float, float]]:
    """
    Detect speech segments using zero-crossing rate and energy.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        frame_length: Length of each frame in samples.
        hop_length: Number of samples between frames.
        zcr_threshold: Zero-crossing rate threshold.
        energy_threshold: Energy threshold.
        min_speech_duration: Minimum duration of speech in seconds.
        
    Returns:
        List of (start_time, end_time) tuples for speech regions.
    """
    logger.info("Performing ZCR-based VAD")
    
    # Calculate zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Calculate energy
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Find frames that are likely to be speech (low ZCR and high energy)
    speech_frames = np.where((zcr < zcr_threshold) & (energy > energy_threshold))[0]
    
    if len(speech_frames) == 0:
        logger.warning("No speech detected")
        return []
    
    # Group consecutive frames
    speech_regions = []
    if len(speech_frames) > 0:
        region_start = speech_frames[0]
        
        for i in range(1, len(speech_frames)):
            if speech_frames[i] > speech_frames[i-1] + 1:
                # End of a region
                region_end = speech_frames[i-1]
                
                # Convert to time
                start_time = librosa.frames_to_time(region_start, sr=sample_rate, hop_length=hop_length)
                end_time = librosa.frames_to_time(region_end, sr=sample_rate, hop_length=hop_length)
                
                # Check if duration is long enough
                if end_time - start_time >= min_speech_duration:
                    speech_regions.append((start_time, end_time))
                
                # Start of a new region
                region_start = speech_frames[i]
        
        # Add the last region
        region_end = speech_frames[-1]
        start_time = librosa.frames_to_time(region_start, sr=sample_rate, hop_length=hop_length)
        end_time = librosa.frames_to_time(region_end, sr=sample_rate, hop_length=hop_length)
        
        if end_time - start_time >= min_speech_duration:
            speech_regions.append((start_time, end_time))
    
    logger.info(f"Detected {len(speech_regions)} speech regions")
    return speech_regions

def apply_vad(audio_data: np.ndarray, sample_rate: int, 
             method: str = "energy", 
             energy_threshold: float = 0.01,
             zcr_threshold: float = 0.2,
             min_speech_duration: float = 0.1) -> np.ndarray:
    """
    Apply Voice Activity Detection to extract only speech segments from audio.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        method: VAD method to use. Options: "energy", "zcr".
        energy_threshold: Energy threshold for speech detection.
        zcr_threshold: Zero-crossing rate threshold for speech detection.
        min_speech_duration: Minimum duration of speech in seconds.
        
    Returns:
        Audio data with only speech segments.
    """
    logger.info(f"Applying VAD using {method} method")
    
    # Detect speech regions
    if method == "energy":
        speech_regions = energy_vad(
            audio_data, 
            sample_rate, 
            energy_threshold=energy_threshold,
            min_speech_duration=min_speech_duration
        )
    elif method == "zcr":
        speech_regions = zcr_vad(
            audio_data, 
            sample_rate, 
            zcr_threshold=zcr_threshold,
            energy_threshold=energy_threshold,
            min_speech_duration=min_speech_duration
        )
    else:
        logger.error(f"Unknown VAD method: {method}")
        return audio_data
    
    if len(speech_regions) == 0:
        logger.warning("No speech regions detected, returning original audio")
        return audio_data
    
    # Extract speech segments
    speech_audio = []
    for start_time, end_time in speech_regions:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        speech_audio.extend(audio_data[start_sample:end_sample])
    
    speech_audio = np.array(speech_audio)
    logger.info(f"Extracted speech audio with shape {speech_audio.shape}")
    
    return speech_audio
