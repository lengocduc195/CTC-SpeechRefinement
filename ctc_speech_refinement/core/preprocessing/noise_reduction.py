"""
Noise reduction module for audio preprocessing.

This module provides functions for reducing noise in audio data
using various techniques.
"""

import numpy as np
import librosa
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import scipy.signal as signal
from scipy.ndimage import median_filter
import noisereduce as nr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def spectral_subtraction(audio_data: np.ndarray, sample_rate: int, 
                        noise_clip: Optional[np.ndarray] = None,
                        n_fft: int = 2048, hop_length: int = 512,
                        noise_frames: int = 10) -> np.ndarray:
    """
    Apply spectral subtraction for noise reduction.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        noise_clip: Noise profile to use. If None, use the first few frames of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        noise_frames: Number of frames to use for noise profile if noise_clip is None.
        
    Returns:
        Noise-reduced audio data.
    """
    logger.info("Applying spectral subtraction")
    
    # Get noise profile
    if noise_clip is None:
        # Use the first few frames as noise profile
        noise_samples = int(noise_frames * hop_length)
        if len(audio_data) <= noise_samples:
            logger.warning("Audio too short for noise estimation, returning original audio")
            return audio_data
        noise_clip = audio_data[:noise_samples]
    
    # Compute STFT of the audio
    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    stft_mag, stft_phase = librosa.magphase(stft)
    
    # Compute STFT of the noise
    noise_stft = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)
    noise_stft_mag = np.abs(noise_stft)
    
    # Estimate noise spectrum
    noise_spectrum = np.mean(noise_stft_mag, axis=1, keepdims=True)
    
    # Apply spectral subtraction
    gain = np.maximum(stft_mag - noise_spectrum, 0) / (stft_mag + 1e-10)
    enhanced_stft = stft * gain
    
    # Inverse STFT
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
    
    # Ensure same length as input
    if len(enhanced_audio) > len(audio_data):
        enhanced_audio = enhanced_audio[:len(audio_data)]
    elif len(enhanced_audio) < len(audio_data):
        enhanced_audio = np.pad(enhanced_audio, (0, len(audio_data) - len(enhanced_audio)))
    
    logger.info("Spectral subtraction completed")
    return enhanced_audio

def wiener_filter(audio_data: np.ndarray, sample_rate: int, 
                 noise_clip: Optional[np.ndarray] = None,
                 noise_frames: int = 10) -> np.ndarray:
    """
    Apply Wiener filter for noise reduction.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        noise_clip: Noise profile to use. If None, use the first few frames of the audio.
        noise_frames: Number of frames to use for noise profile if noise_clip is None.
        
    Returns:
        Noise-reduced audio data.
    """
    logger.info("Applying Wiener filter")
    
    # Get noise profile
    if noise_clip is None:
        # Use the first few frames as noise profile
        noise_samples = int(noise_frames * sample_rate / 100)  # Convert frames to samples
        if len(audio_data) <= noise_samples:
            logger.warning("Audio too short for noise estimation, returning original audio")
            return audio_data
        noise_clip = audio_data[:noise_samples]
    
    # Apply Wiener filter using scipy
    enhanced_audio = signal.wiener(audio_data)
    
    logger.info("Wiener filtering completed")
    return enhanced_audio

def median_filtering(audio_data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filtering for noise reduction.
    
    Args:
        audio_data: Audio data as numpy array.
        kernel_size: Size of the median filter kernel.
        
    Returns:
        Noise-reduced audio data.
    """
    logger.info(f"Applying median filtering with kernel size {kernel_size}")
    
    # Apply median filter
    enhanced_audio = median_filter(audio_data, size=kernel_size)
    
    logger.info("Median filtering completed")
    return enhanced_audio

def noisereduce_lib(audio_data: np.ndarray, sample_rate: int, 
                   noise_clip: Optional[np.ndarray] = None,
                   noise_frames: int = 10,
                   stationary: bool = True,
                   prop_decrease: float = 0.75) -> np.ndarray:
    """
    Apply noise reduction using the noisereduce library.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        noise_clip: Noise profile to use. If None, use the first few frames of the audio.
        noise_frames: Number of frames to use for noise profile if noise_clip is None.
        stationary: Whether the noise is stationary.
        prop_decrease: Proportion of noise to remove.
        
    Returns:
        Noise-reduced audio data.
    """
    logger.info("Applying noise reduction using noisereduce library")
    
    try:
        # Get noise profile
        if noise_clip is None:
            # Use the first few frames as noise profile
            noise_samples = int(noise_frames * sample_rate / 10)  # Convert frames to samples
            if len(audio_data) <= noise_samples:
                logger.warning("Audio too short for noise estimation, using non-stationary mode")
                enhanced_audio = nr.reduce_noise(
                    y=audio_data, 
                    sr=sample_rate,
                    stationary=False,
                    prop_decrease=prop_decrease
                )
            else:
                noise_clip = audio_data[:noise_samples]
                enhanced_audio = nr.reduce_noise(
                    y=audio_data, 
                    sr=sample_rate,
                    y_noise=noise_clip,
                    stationary=stationary,
                    prop_decrease=prop_decrease
                )
        else:
            enhanced_audio = nr.reduce_noise(
                y=audio_data, 
                sr=sample_rate,
                y_noise=noise_clip,
                stationary=stationary,
                prop_decrease=prop_decrease
            )
        
        logger.info("Noise reduction completed")
        return enhanced_audio
    
    except Exception as e:
        logger.error(f"Error in noise reduction: {str(e)}")
        logger.warning("Returning original audio")
        return audio_data

def reduce_noise(audio_data: np.ndarray, sample_rate: int, 
                method: str = "spectral_subtraction",
                noise_clip: Optional[np.ndarray] = None,
                noise_frames: int = 10,
                kernel_size: int = 3,
                stationary: bool = True,
                prop_decrease: float = 0.75) -> np.ndarray:
    """
    Apply noise reduction to audio data using the specified method.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        method: Noise reduction method. Options: "spectral_subtraction", "wiener", "median", "noisereduce".
        noise_clip: Noise profile to use. If None, use the first few frames of the audio.
        noise_frames: Number of frames to use for noise profile if noise_clip is None.
        kernel_size: Size of the median filter kernel (for median filtering).
        stationary: Whether the noise is stationary (for noisereduce method).
        prop_decrease: Proportion of noise to remove (for noisereduce method).
        
    Returns:
        Noise-reduced audio data.
    """
    logger.info(f"Reducing noise using {method} method")
    
    if method == "spectral_subtraction":
        return spectral_subtraction(audio_data, sample_rate, noise_clip, noise_frames=noise_frames)
    elif method == "wiener":
        return wiener_filter(audio_data, sample_rate, noise_clip, noise_frames=noise_frames)
    elif method == "median":
        return median_filtering(audio_data, kernel_size=kernel_size)
    elif method == "noisereduce":
        return noisereduce_lib(
            audio_data, 
            sample_rate, 
            noise_clip, 
            noise_frames=noise_frames,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
    else:
        logger.error(f"Unknown noise reduction method: {method}")
        return audio_data
