"""
Preprocessing utilities for audio EDA.
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path
from scipy import signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio data to a target sample rate.
    
    Args:
        audio_data: Audio data as numpy array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio data.
    """
    logger.info(f"Resampling audio from {orig_sr} Hz to {target_sr} Hz")
    return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)

def trim_audio(audio_data: np.ndarray, sample_rate: int, start_time: float = 0.0, 
              end_time: Optional[float] = None) -> np.ndarray:
    """
    Trim audio data to a specific time range.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        start_time: Start time in seconds.
        end_time: End time in seconds. If None, trim to the end of the audio.
        
    Returns:
        Trimmed audio data.
    """
    logger.info(f"Trimming audio from {start_time} s to {end_time if end_time else 'end'}")
    
    start_sample = int(start_time * sample_rate)
    
    if end_time is not None:
        end_sample = int(end_time * sample_rate)
        return audio_data[start_sample:end_sample]
    else:
        return audio_data[start_sample:]

def apply_fade(audio_data: np.ndarray, sample_rate: int, fade_in_time: float = 0.0, 
              fade_out_time: float = 0.0) -> np.ndarray:
    """
    Apply fade-in and fade-out to audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        fade_in_time: Fade-in time in seconds.
        fade_out_time: Fade-out time in seconds.
        
    Returns:
        Audio data with fades applied.
    """
    logger.info(f"Applying fade-in of {fade_in_time} s and fade-out of {fade_out_time} s")
    
    # Create a copy of the audio data
    audio_with_fade = audio_data.copy()
    
    # Apply fade-in
    if fade_in_time > 0:
        fade_in_samples = int(fade_in_time * sample_rate)
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        audio_with_fade[:fade_in_samples] *= fade_in_curve
    
    # Apply fade-out
    if fade_out_time > 0:
        fade_out_samples = int(fade_out_time * sample_rate)
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        audio_with_fade[-fade_out_samples:] *= fade_out_curve
    
    return audio_with_fade

def apply_filter(audio_data: np.ndarray, sample_rate: int, filter_type: str = "lowpass", 
               cutoff_freq: Union[float, Tuple[float, float]] = 1000.0, 
               order: int = 4) -> np.ndarray:
    """
    Apply a filter to audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        filter_type: Type of filter. Options: "lowpass", "highpass", "bandpass", "bandstop".
        cutoff_freq: Cutoff frequency in Hz. For bandpass and bandstop, provide a tuple of (low, high).
        order: Filter order.
        
    Returns:
        Filtered audio data.
    """
    logger.info(f"Applying {filter_type} filter with cutoff {cutoff_freq} Hz")
    
    nyquist = 0.5 * sample_rate
    
    if filter_type == "lowpass":
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low')
    elif filter_type == "highpass":
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='high')
    elif filter_type == "bandpass":
        low, high = cutoff_freq
        low_normal = low / nyquist
        high_normal = high / nyquist
        b, a = signal.butter(order, [low_normal, high_normal], btype='band')
    elif filter_type == "bandstop":
        low, high = cutoff_freq
        low_normal = low / nyquist
        high_normal = high / nyquist
        b, a = signal.butter(order, [low_normal, high_normal], btype='bandstop')
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")
    
    filtered_audio = signal.filtfilt(b, a, audio_data)
    return filtered_audio

def add_noise(audio_data: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """
    Add Gaussian noise to audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        noise_level: Standard deviation of the noise.
        
    Returns:
        Audio data with added noise.
    """
    logger.info(f"Adding Gaussian noise with level {noise_level}")
    
    noise = np.random.normal(0, noise_level, len(audio_data))
    return audio_data + noise

def time_stretch(audio_data: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Apply time stretching to audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        rate: Stretching rate. Values > 1 speed up, values < 1 slow down.
        
    Returns:
        Time-stretched audio data.
    """
    logger.info(f"Applying time stretching with rate {rate}")
    
    return librosa.effects.time_stretch(audio_data, rate=rate)

def pitch_shift(audio_data: np.ndarray, sample_rate: int, n_steps: float = 0.0) -> np.ndarray:
    """
    Apply pitch shifting to audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_steps: Number of semitones to shift. Positive values shift up, negative values shift down.
        
    Returns:
        Pitch-shifted audio data.
    """
    logger.info(f"Applying pitch shifting by {n_steps} semitones")
    
    return librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=n_steps)

def preprocess_for_eda(audio_data: np.ndarray, sample_rate: int, 
                      normalize: bool = True, remove_silence: bool = False, 
                      trim_start: float = 0.0, trim_end: Optional[float] = None, 
                      fade_in: float = 0.0, fade_out: float = 0.0, 
                      filter_type: Optional[str] = None, cutoff_freq: Union[float, Tuple[float, float]] = 1000.0, 
                      add_noise_level: Optional[float] = None) -> np.ndarray:
    """
    Apply a series of preprocessing steps to audio data for EDA.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        normalize: Whether to normalize the audio.
        remove_silence: Whether to remove silence from the audio.
        trim_start: Start time for trimming in seconds.
        trim_end: End time for trimming in seconds. If None, trim to the end of the audio.
        fade_in: Fade-in time in seconds.
        fade_out: Fade-out time in seconds.
        filter_type: Type of filter to apply. If None, no filter is applied.
        cutoff_freq: Cutoff frequency for the filter in Hz.
        add_noise_level: Level of noise to add. If None, no noise is added.
        
    Returns:
        Preprocessed audio data.
    """
    logger.info("Applying preprocessing steps for EDA")
    
    # Make a copy of the audio data
    processed_audio = audio_data.copy()
    
    # Apply trimming
    if trim_start > 0 or trim_end is not None:
        processed_audio = trim_audio(processed_audio, sample_rate, trim_start, trim_end)
    
    # Apply normalization
    if normalize:
        mean = np.mean(processed_audio)
        std = np.std(processed_audio)
        if std > 0:
            processed_audio = (processed_audio - mean) / std
        else:
            processed_audio = processed_audio - mean
    
    # Remove silence
    if remove_silence:
        non_silent_intervals = librosa.effects.split(processed_audio, top_db=60)
        if len(non_silent_intervals) > 0:
            non_silent_audio = []
            for interval in non_silent_intervals:
                start, end = interval
                non_silent_audio.extend(processed_audio[start:end])
            processed_audio = np.array(non_silent_audio)
    
    # Apply fades
    if fade_in > 0 or fade_out > 0:
        processed_audio = apply_fade(processed_audio, sample_rate, fade_in, fade_out)
    
    # Apply filter
    if filter_type is not None:
        processed_audio = apply_filter(processed_audio, sample_rate, filter_type, cutoff_freq)
    
    # Add noise
    if add_noise_level is not None:
        processed_audio = add_noise(processed_audio, add_noise_level)
    
    return processed_audio

def batch_preprocess_for_eda(audio_data_dict: Dict[str, Tuple[np.ndarray, int]], 
                            normalize: bool = True, remove_silence: bool = False, 
                            trim_start: float = 0.0, trim_end: Optional[float] = None, 
                            fade_in: float = 0.0, fade_out: float = 0.0, 
                            filter_type: Optional[str] = None, cutoff_freq: Union[float, Tuple[float, float]] = 1000.0, 
                            add_noise_level: Optional[float] = None) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Apply preprocessing steps to a batch of audio files for EDA.
    
    Args:
        audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
        normalize: Whether to normalize the audio.
        remove_silence: Whether to remove silence from the audio.
        trim_start: Start time for trimming in seconds.
        trim_end: End time for trimming in seconds. If None, trim to the end of the audio.
        fade_in: Fade-in time in seconds.
        fade_out: Fade-out time in seconds.
        filter_type: Type of filter to apply. If None, no filter is applied.
        cutoff_freq: Cutoff frequency for the filter in Hz.
        add_noise_level: Level of noise to add. If None, no noise is added.
        
    Returns:
        Dictionary mapping file paths to tuples of (preprocessed_audio, sample_rate).
    """
    logger.info(f"Batch preprocessing {len(audio_data_dict)} files for EDA")
    
    preprocessed_dict = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            preprocessed_audio = preprocess_for_eda(
                audio_data, 
                sample_rate, 
                normalize, 
                remove_silence, 
                trim_start, 
                trim_end, 
                fade_in, 
                fade_out, 
                filter_type, 
                cutoff_freq, 
                add_noise_level
            )
            
            preprocessed_dict[file_path] = (preprocessed_audio, sample_rate)
            logger.info(f"Preprocessed {file_path} for EDA")
            
        except Exception as e:
            logger.error(f"Error preprocessing {file_path} for EDA: {str(e)}")
    
    return preprocessed_dict
