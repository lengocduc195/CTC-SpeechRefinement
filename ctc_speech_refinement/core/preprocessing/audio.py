"""
Audio preprocessing module for CTC Speech Transcription.

This module provides functions for preprocessing audio data, including loading,
normalization, silence removal, noise reduction, frequency normalization, and
Voice Activity Detection (VAD).
"""

import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, List, Dict, Union, Any
import logging

from ctc_speech_refinement.config.config import SAMPLE_RATE, AUDIO_DURATION
from ctc_speech_refinement.core.preprocessing.vad import apply_vad
from ctc_speech_refinement.core.preprocessing.noise_reduction import reduce_noise
from ctc_speech_refinement.core.preprocessing.frequency_normalization import normalize_frequency

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: Optional[float] = AUDIO_DURATION) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample it to the target sample rate.

    Args:
        file_path: Path to the audio file.
        sr: Target sample rate.
        duration: Duration in seconds to load. If None, load the entire file.

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        logger.info(f"Loading audio file: {file_path}")
        audio_data, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
        logger.info(f"Loaded audio with shape {audio_data.shape} and sample rate {sample_rate}")
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        raise

def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalize audio data to have zero mean and unit variance.

    Args:
        audio_data: Audio data as numpy array.

    Returns:
        Normalized audio data.
    """
    logger.info("Normalizing audio data")
    mean = np.mean(audio_data)
    std = np.std(audio_data)
    if std > 0:
        normalized_audio = (audio_data - mean) / std
    else:
        normalized_audio = audio_data - mean
    return normalized_audio

def remove_silence(audio_data: np.ndarray, sample_rate: int,
                  top_db: int = 60, frame_length: int = 2048,
                  hop_length: int = 512) -> np.ndarray:
    """
    Remove silence from audio data.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        top_db: Threshold for silence detection in dB.
        frame_length: Length of each frame in samples.
        hop_length: Number of samples between frames.

    Returns:
        Audio data with silence removed.
    """
    logger.info("Removing silence from audio")
    non_silent_intervals = librosa.effects.split(audio_data,
                                               top_db=top_db,
                                               frame_length=frame_length,
                                               hop_length=hop_length)

    if len(non_silent_intervals) == 0:
        logger.warning("No non-silent intervals found")
        return audio_data

    non_silent_audio = []
    for interval in non_silent_intervals:
        start, end = interval
        non_silent_audio.extend(audio_data[start:end])

    return np.array(non_silent_audio)

def preprocess_audio(file_path: str, normalize: bool = True,
                    remove_silence_flag: bool = True,
                    apply_vad_flag: bool = False, vad_method: str = "energy",
                    reduce_noise_flag: bool = False, noise_reduction_method: str = "spectral_subtraction",
                    normalize_frequency_flag: bool = False, frequency_normalization_method: str = "bandpass") -> Tuple[np.ndarray, int]:
    """
    Preprocess audio file by loading, normalizing, removing silence, applying VAD,
    reducing noise, and normalizing frequency.

    Args:
        file_path: Path to the audio file.
        normalize: Whether to normalize the audio.
        remove_silence_flag: Whether to remove silence from the audio.
        apply_vad_flag: Whether to apply Voice Activity Detection.
        vad_method: VAD method to use. Options: "energy", "zcr".
        reduce_noise_flag: Whether to apply noise reduction.
        noise_reduction_method: Noise reduction method. Options: "spectral_subtraction", "wiener", "median", "noisereduce".
        normalize_frequency_flag: Whether to normalize frequency content.
        frequency_normalization_method: Frequency normalization method. Options: "bandpass", "preemphasis", "equalize", "combined".

    Returns:
        Tuple of (preprocessed_audio, sample_rate)
    """
    logger.info(f"Preprocessing audio file: {file_path}")
    audio_data, sample_rate = load_audio(file_path)

    # Apply preprocessing steps in a sensible order

    # 1. Normalize frequency first (if requested)
    if normalize_frequency_flag:
        logger.info(f"Applying frequency normalization with method: {frequency_normalization_method}")
        audio_data = normalize_frequency(audio_data, sample_rate, method=frequency_normalization_method)

    # 2. Apply noise reduction (if requested)
    if reduce_noise_flag:
        logger.info(f"Applying noise reduction with method: {noise_reduction_method}")
        audio_data = reduce_noise(audio_data, sample_rate, method=noise_reduction_method)

    # 3. Apply VAD (if requested)
    if apply_vad_flag:
        logger.info(f"Applying Voice Activity Detection with method: {vad_method}")
        audio_data = apply_vad(audio_data, sample_rate, method=vad_method)

    # 4. Remove silence (if requested and VAD not applied)
    if remove_silence_flag and not apply_vad_flag:
        audio_data = remove_silence(audio_data, sample_rate)

    # 5. Normalize amplitude (if requested)
    if normalize:
        audio_data = normalize_audio(audio_data)

    logger.info(f"Preprocessed audio shape: {audio_data.shape}")
    return audio_data, sample_rate

def batch_preprocess(file_paths: List[str], output_dir: Optional[str] = None,
                    normalize: bool = True, remove_silence_flag: bool = True,
                    apply_vad_flag: bool = False, vad_method: str = "energy",
                    reduce_noise_flag: bool = False, noise_reduction_method: str = "spectral_subtraction",
                    normalize_frequency_flag: bool = False, frequency_normalization_method: str = "bandpass") -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Preprocess a batch of audio files.

    Args:
        file_paths: List of paths to audio files.
        output_dir: Directory to save preprocessed audio files. If None, don't save.
        normalize: Whether to normalize the audio.
        remove_silence_flag: Whether to remove silence from the audio.
        apply_vad_flag: Whether to apply Voice Activity Detection.
        vad_method: VAD method to use. Options: "energy", "zcr".
        reduce_noise_flag: Whether to apply noise reduction.
        noise_reduction_method: Noise reduction method. Options: "spectral_subtraction", "wiener", "median", "noisereduce".
        normalize_frequency_flag: Whether to normalize frequency content.
        frequency_normalization_method: Frequency normalization method. Options: "bandpass", "preemphasis", "equalize", "combined".

    Returns:
        Dictionary mapping file paths to tuples of (preprocessed_audio, sample_rate)
    """
    logger.info(f"Batch preprocessing {len(file_paths)} audio files")
    results = {}

    for file_path in file_paths:
        try:
            audio_data, sample_rate = preprocess_audio(
                file_path,
                normalize=normalize,
                remove_silence_flag=remove_silence_flag,
                apply_vad_flag=apply_vad_flag,
                vad_method=vad_method,
                reduce_noise_flag=reduce_noise_flag,
                noise_reduction_method=noise_reduction_method,
                normalize_frequency_flag=normalize_frequency_flag,
                frequency_normalization_method=frequency_normalization_method
            )
            results[file_path] = (audio_data, sample_rate)

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                sf.write(output_path, audio_data, sample_rate)
                logger.info(f"Saved preprocessed audio to {output_path}")

        except Exception as e:
            logger.error(f"Error preprocessing {file_path}: {str(e)}")

    return results
