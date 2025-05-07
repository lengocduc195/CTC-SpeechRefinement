"""
Spectral features module for audio analysis.

This module provides functions for computing various spectral features of audio data.
"""

import numpy as np
import librosa
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_stft(audio_data: np.ndarray, n_fft: int = 2048, hop_length: int = 512, 
                win_length: Optional[int] = None) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        win_length: Window length in samples. If None, defaults to n_fft.
        
    Returns:
        STFT of audio data.
    """
    logger.info(f"Computing STFT with n_fft={n_fft}, hop_length={hop_length}")
    
    stft = librosa.stft(
        y=audio_data, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length
    )
    
    logger.info(f"STFT shape: {stft.shape}")
    return stft

def compute_mel_spectrogram(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                           hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
    """
    Compute the Mel spectrogram of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        n_mels: Number of Mel bands.
        
    Returns:
        Mel spectrogram of audio data.
    """
    logger.info(f"Computing Mel spectrogram with n_mels={n_mels}")
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
    return mel_spec

def compute_spectral_centroid(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                             hop_length: int = 512) -> np.ndarray:
    """
    Compute the spectral centroid of audio data.
    
    The spectral centroid indicates where the "center of mass" of the spectrum is located.
    It is a measure of the brightness of a sound.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        
    Returns:
        Spectral centroid of audio data.
    """
    logger.info("Computing spectral centroid")
    
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length
    )[0]
    
    logger.info(f"Spectral centroid shape: {spectral_centroid.shape}")
    return spectral_centroid

def compute_spectral_bandwidth(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                              hop_length: int = 512) -> np.ndarray:
    """
    Compute the spectral bandwidth of audio data.
    
    The spectral bandwidth measures the width of the spectrum around its centroid.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        
    Returns:
        Spectral bandwidth of audio data.
    """
    logger.info("Computing spectral bandwidth")
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length
    )[0]
    
    logger.info(f"Spectral bandwidth shape: {spectral_bandwidth.shape}")
    return spectral_bandwidth

def compute_spectral_contrast(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                             hop_length: int = 512, n_bands: int = 6) -> np.ndarray:
    """
    Compute the spectral contrast of audio data.
    
    Spectral contrast measures the difference between peaks and valleys in the spectrum.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        n_bands: Number of frequency bands.
        
    Returns:
        Spectral contrast of audio data.
    """
    logger.info(f"Computing spectral contrast with n_bands={n_bands}")
    
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_bands=n_bands
    )
    
    logger.info(f"Spectral contrast shape: {spectral_contrast.shape}")
    return spectral_contrast

def compute_spectral_flatness(audio_data: np.ndarray, n_fft: int = 2048, 
                             hop_length: int = 512) -> np.ndarray:
    """
    Compute the spectral flatness of audio data.
    
    Spectral flatness measures how noise-like a sound is, as opposed to being tone-like.
    
    Args:
        audio_data: Audio data as numpy array.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        
    Returns:
        Spectral flatness of audio data.
    """
    logger.info("Computing spectral flatness")
    
    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio_data, 
        n_fft=n_fft, 
        hop_length=hop_length
    )[0]
    
    logger.info(f"Spectral flatness shape: {spectral_flatness.shape}")
    return spectral_flatness

def compute_spectral_rolloff(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                            hop_length: int = 512, roll_percent: float = 0.85) -> np.ndarray:
    """
    Compute the spectral rolloff of audio data.
    
    Spectral rolloff is the frequency below which a specified percentage of the total spectral energy lies.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        roll_percent: Roll-off percentage.
        
    Returns:
        Spectral rolloff of audio data.
    """
    logger.info(f"Computing spectral rolloff with roll_percent={roll_percent}")
    
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length,
        roll_percent=roll_percent
    )[0]
    
    logger.info(f"Spectral rolloff shape: {spectral_rolloff.shape}")
    return spectral_rolloff

def compute_all_spectral_features(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                                 hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    Compute all spectral features of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        
    Returns:
        Dictionary of spectral features.
    """
    logger.info("Computing all spectral features")
    
    features = {
        "stft": compute_stft(audio_data, n_fft, hop_length),
        "mel_spectrogram": compute_mel_spectrogram(audio_data, sample_rate, n_fft, hop_length),
        "spectral_centroid": compute_spectral_centroid(audio_data, sample_rate, n_fft, hop_length),
        "spectral_bandwidth": compute_spectral_bandwidth(audio_data, sample_rate, n_fft, hop_length),
        "spectral_contrast": compute_spectral_contrast(audio_data, sample_rate, n_fft, hop_length),
        "spectral_flatness": compute_spectral_flatness(audio_data, n_fft, hop_length),
        "spectral_rolloff": compute_spectral_rolloff(audio_data, sample_rate, n_fft, hop_length)
    }
    
    return features
