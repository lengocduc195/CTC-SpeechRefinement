"""
Feature extraction module for CTC Speech Transcription.
"""

import numpy as np
import librosa
import logging
from typing import Dict, Tuple, List, Optional, Union

from config.config import (
    FEATURE_TYPE, N_MFCC, N_MELS, N_FFT, 
    HOP_LENGTH, WIN_LENGTH, SAMPLE_RATE
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_mfcc(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE, 
                n_mfcc: int = N_MFCC, n_fft: int = N_FFT, 
                hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH) -> np.ndarray:
    """
    Extract MFCC features from audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_mfcc: Number of MFCC coefficients to extract.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        win_length: Window length in samples.
        
    Returns:
        MFCC features as numpy array of shape (n_mfcc, time).
    """
    logger.info(f"Extracting {n_mfcc} MFCC features")
    mfccs = librosa.feature.mfcc(
        y=audio_data, 
        sr=sample_rate, 
        n_mfcc=n_mfcc, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length
    )
    logger.info(f"MFCC features shape: {mfccs.shape}")
    return mfccs

def extract_mel_spectrogram(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE, 
                           n_mels: int = N_MELS, n_fft: int = N_FFT, 
                           hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH) -> np.ndarray:
    """
    Extract Mel spectrogram features from audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_mels: Number of Mel bands.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        win_length: Window length in samples.
        
    Returns:
        Mel spectrogram features as numpy array of shape (n_mels, time).
    """
    logger.info(f"Extracting Mel spectrogram with {n_mels} bands")
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, 
        sr=sample_rate, 
        n_mels=n_mels, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length
    )
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    logger.info(f"Mel spectrogram features shape: {log_mel_spec.shape}")
    return log_mel_spec

def extract_spectrogram(audio_data: np.ndarray, n_fft: int = N_FFT, 
                       hop_length: int = HOP_LENGTH, win_length: int = WIN_LENGTH) -> np.ndarray:
    """
    Extract spectrogram features from audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        win_length: Window length in samples.
        
    Returns:
        Spectrogram features as numpy array.
    """
    logger.info("Extracting spectrogram")
    spec = np.abs(librosa.stft(
        y=audio_data, 
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=win_length
    ))
    # Convert to log scale (dB)
    log_spec = librosa.amplitude_to_db(spec, ref=np.max)
    logger.info(f"Spectrogram features shape: {log_spec.shape}")
    return log_spec

def extract_features(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE, 
                    feature_type: str = FEATURE_TYPE) -> np.ndarray:
    """
    Extract features from audio data based on the specified feature type.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        feature_type: Type of features to extract. Options: "mfcc", "mel_spectrogram", "spectrogram".
        
    Returns:
        Extracted features as numpy array.
    """
    logger.info(f"Extracting features of type: {feature_type}")
    
    if feature_type == "mfcc":
        features = extract_mfcc(audio_data, sample_rate)
    elif feature_type == "mel_spectrogram":
        features = extract_mel_spectrogram(audio_data, sample_rate)
    elif feature_type == "spectrogram":
        features = extract_spectrogram(audio_data)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    return features

def normalize_features(features: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalize features to have zero mean and unit variance.
    
    Args:
        features: Features as numpy array.
        axis: Axis along which to normalize.
        
    Returns:
        Normalized features.
    """
    logger.info(f"Normalizing features along axis {axis}")
    mean = np.mean(features, axis=axis, keepdims=True)
    std = np.std(features, axis=axis, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    normalized_features = (features - mean) / std
    return normalized_features

def pad_features(features: np.ndarray, max_length: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad features to a fixed length.
    
    Args:
        features: Features as numpy array of shape (feature_dim, time).
        max_length: Maximum length to pad to.
        pad_value: Value to use for padding.
        
    Returns:
        Padded features as numpy array of shape (feature_dim, max_length).
    """
    logger.info(f"Padding features to length {max_length}")
    feature_dim, time_steps = features.shape
    
    if time_steps >= max_length:
        return features[:, :max_length]
    
    padding = np.full((feature_dim, max_length - time_steps), pad_value)
    padded_features = np.concatenate([features, padding], axis=1)
    return padded_features

def batch_extract_features(audio_data_dict: Dict[str, Tuple[np.ndarray, int]], 
                          feature_type: str = FEATURE_TYPE, 
                          normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    Extract features from a batch of audio data.
    
    Args:
        audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
        feature_type: Type of features to extract.
        normalize: Whether to normalize the features.
        
    Returns:
        Dictionary mapping file paths to extracted features.
    """
    logger.info(f"Batch extracting {feature_type} features for {len(audio_data_dict)} files")
    features_dict = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            features = extract_features(audio_data, sample_rate, feature_type)
            
            if normalize:
                features = normalize_features(features)
                
            features_dict[file_path] = features
            logger.info(f"Extracted features for {file_path} with shape {features.shape}")
            
        except Exception as e:
            logger.error(f"Error extracting features for {file_path}: {str(e)}")
    
    return features_dict
