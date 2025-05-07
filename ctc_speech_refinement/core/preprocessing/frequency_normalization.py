"""
Frequency normalization module for audio preprocessing.

This module provides functions for normalizing the frequency content of audio data
using various techniques.
"""

import numpy as np
import librosa
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import scipy.signal as signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def equalize_spectrum(audio_data: np.ndarray, sample_rate: int, 
                     n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Apply spectral equalization to flatten the frequency spectrum.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        
    Returns:
        Frequency-normalized audio data.
    """
    logger.info("Applying spectral equalization")
    
    # Compute STFT
    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    stft_mag, stft_phase = librosa.magphase(stft)
    
    # Compute average spectrum
    avg_spectrum = np.mean(stft_mag, axis=1, keepdims=True)
    
    # Normalize each frame by the average spectrum
    gain = 1.0 / (avg_spectrum + 1e-10)
    normalized_stft = stft * gain
    
    # Inverse STFT
    normalized_audio = librosa.istft(normalized_stft, hop_length=hop_length)
    
    # Ensure same length as input
    if len(normalized_audio) > len(audio_data):
        normalized_audio = normalized_audio[:len(audio_data)]
    elif len(normalized_audio) < len(audio_data):
        normalized_audio = np.pad(normalized_audio, (0, len(audio_data) - len(normalized_audio)))
    
    logger.info("Spectral equalization completed")
    return normalized_audio

def apply_bandpass_filter(audio_data: np.ndarray, sample_rate: int, 
                         low_freq: float = 80.0, high_freq: float = 8000.0,
                         order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to focus on speech frequency range.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        low_freq: Lower cutoff frequency in Hz.
        high_freq: Upper cutoff frequency in Hz.
        order: Filter order.
        
    Returns:
        Filtered audio data.
    """
    logger.info(f"Applying bandpass filter ({low_freq}-{high_freq} Hz)")
    
    # Normalize frequencies
    nyquist = 0.5 * sample_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    logger.info("Bandpass filtering completed")
    return filtered_audio

def apply_preemphasis(audio_data: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to boost high frequencies.
    
    Args:
        audio_data: Audio data as numpy array.
        coef: Pre-emphasis coefficient.
        
    Returns:
        Pre-emphasized audio data.
    """
    logger.info(f"Applying pre-emphasis with coefficient {coef}")
    
    # Apply pre-emphasis filter: y[n] = x[n] - coef * x[n-1]
    emphasized_audio = np.append(audio_data[0], audio_data[1:] - coef * audio_data[:-1])
    
    logger.info("Pre-emphasis completed")
    return emphasized_audio

def normalize_frequency(audio_data: np.ndarray, sample_rate: int, 
                       method: str = "bandpass",
                       low_freq: float = 80.0, high_freq: float = 8000.0,
                       preemphasis_coef: float = 0.97,
                       n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Normalize frequency content of audio data using the specified method.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        method: Frequency normalization method. Options: "bandpass", "preemphasis", "equalize", "combined".
        low_freq: Lower cutoff frequency for bandpass filter in Hz.
        high_freq: Upper cutoff frequency for bandpass filter in Hz.
        preemphasis_coef: Pre-emphasis coefficient.
        n_fft: FFT window size for spectral equalization.
        hop_length: Number of samples between frames for spectral equalization.
        
    Returns:
        Frequency-normalized audio data.
    """
    logger.info(f"Normalizing frequency using {method} method")
    
    if method == "bandpass":
        return apply_bandpass_filter(audio_data, sample_rate, low_freq, high_freq)
    elif method == "preemphasis":
        return apply_preemphasis(audio_data, preemphasis_coef)
    elif method == "equalize":
        return equalize_spectrum(audio_data, sample_rate, n_fft, hop_length)
    elif method == "combined":
        # Apply bandpass filter first
        filtered_audio = apply_bandpass_filter(audio_data, sample_rate, low_freq, high_freq)
        
        # Then apply pre-emphasis
        emphasized_audio = apply_preemphasis(filtered_audio, preemphasis_coef)
        
        # Finally apply spectral equalization
        normalized_audio = equalize_spectrum(emphasized_audio, sample_rate, n_fft, hop_length)
        
        return normalized_audio
    else:
        logger.error(f"Unknown frequency normalization method: {method}")
        return audio_data
