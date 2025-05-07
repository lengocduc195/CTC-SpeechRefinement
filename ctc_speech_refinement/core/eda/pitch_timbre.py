"""
Pitch and timbre analysis for audio data.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def estimate_pitch(audio_data: np.ndarray, sample_rate: int, frame_length: int = 2048, 
                  hop_length: int = 512, fmin: float = 50.0, fmax: float = 2000.0) -> np.ndarray:
    """
    Estimate pitch (fundamental frequency) of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        frame_length: Length of each frame in samples.
        hop_length: Number of samples between frames.
        fmin: Minimum frequency to consider.
        fmax: Maximum frequency to consider.
        
    Returns:
        Array of pitch values over time.
    """
    logger.info("Estimating pitch")
    
    # Estimate pitch using PYIN algorithm
    pitch, voiced_flag, voiced_probs = librosa.pyin(
        audio_data, 
        fmin=fmin, 
        fmax=fmax, 
        sr=sample_rate, 
        frame_length=frame_length, 
        hop_length=hop_length
    )
    
    return pitch

def plot_pitch(pitch: np.ndarray, sample_rate: int, hop_length: int = 512, 
              title: str = "Pitch Estimation", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot pitch estimation over time.
    
    Args:
        pitch: Array of pitch values.
        sample_rate: Sample rate of the audio.
        hop_length: Number of samples between frames.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting pitch estimation")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create time array
    frames = range(len(pitch))
    t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
    
    # Plot pitch
    ax.plot(t, pitch, color='blue', alpha=0.7)
    
    # Add horizontal line for mean pitch (excluding NaN values)
    valid_pitch = pitch[~np.isnan(pitch)]
    if len(valid_pitch) > 0:
        mean_pitch = np.mean(valid_pitch)
        ax.axhline(mean_pitch, color='r', linestyle='--', linewidth=1, 
                  label=f'Mean Pitch: {mean_pitch:.1f} Hz')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to log scale for better visualization
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved pitch plot to {save_path}")
    
    return fig

def plot_pitch_distribution(pitch: np.ndarray, title: str = "Pitch Distribution", 
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of pitch values.
    
    Args:
        pitch: Array of pitch values.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting pitch distribution")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out NaN values
    valid_pitch = pitch[~np.isnan(pitch)]
    
    if len(valid_pitch) > 0:
        # Plot histogram
        ax.hist(valid_pitch, bins=50, alpha=0.7, color='blue')
        
        # Add vertical line for mean pitch
        mean_pitch = np.mean(valid_pitch)
        ax.axvline(mean_pitch, color='r', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_pitch:.1f} Hz')
        
        # Add vertical line for median pitch
        median_pitch = np.median(valid_pitch)
        ax.axvline(median_pitch, color='g', linestyle='--', linewidth=2, 
                  label=f'Median: {median_pitch:.1f} Hz')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to log scale for better visualization
        ax.set_xscale('log')
    else:
        ax.text(0.5, 0.5, "No valid pitch data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved pitch distribution plot to {save_path}")
    
    return fig

def extract_mfcc_for_timbre(audio_data: np.ndarray, sample_rate: int, n_mfcc: int = 13, 
                           n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Extract MFCC features for timbre analysis.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        
    Returns:
        MFCC features.
    """
    logger.info(f"Extracting {n_mfcc} MFCC features for timbre analysis")
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(
        y=audio_data, 
        sr=sample_rate, 
        n_mfcc=n_mfcc, 
        n_fft=n_fft, 
        hop_length=hop_length
    )
    
    return mfccs

def plot_mfcc(mfccs: np.ndarray, sample_rate: int, hop_length: int = 512, 
             title: str = "MFCC Features", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot MFCC features.
    
    Args:
        mfccs: MFCC features.
        sample_rate: Sample rate of the audio.
        hop_length: Number of samples between frames.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting MFCC features")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot MFCC
    img = librosa.display.specshow(
        mfccs, 
        sr=sample_rate, 
        hop_length=hop_length, 
        x_axis='time',
        ax=ax
    )
    
    fig.colorbar(img, ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('MFCC Coefficients')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved MFCC plot to {save_path}")
    
    return fig

def plot_mfcc_statistics(mfccs: np.ndarray, title: str = "MFCC Statistics", 
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot statistics of MFCC features.
    
    Args:
        mfccs: MFCC features.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting MFCC statistics")
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate mean and standard deviation for each coefficient
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    
    # Plot mean values
    axs[0].bar(range(len(mfcc_means)), mfcc_means, alpha=0.7, color='blue')
    axs[0].set_xlabel('MFCC Coefficient')
    axs[0].set_ylabel('Mean Value')
    axs[0].set_title('Mean MFCC Values')
    axs[0].grid(True, alpha=0.3)
    
    # Plot standard deviation values
    axs[1].bar(range(len(mfcc_stds)), mfcc_stds, alpha=0.7, color='green')
    axs[1].set_xlabel('MFCC Coefficient')
    axs[1].set_ylabel('Standard Deviation')
    axs[1].set_title('MFCC Standard Deviations')
    axs[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved MFCC statistics plot to {save_path}")
    
    return fig

def analyze_pitch_timbre(audio_data: np.ndarray, sample_rate: int, 
                        title_prefix: str = "", output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive pitch and timbre analysis on audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        title_prefix: Prefix for plot titles.
        output_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary containing analysis results and figure objects.
    """
    logger.info("Performing pitch and timbre analysis")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Estimate pitch
    pitch = estimate_pitch(audio_data, sample_rate)
    
    # Extract MFCC features for timbre analysis
    mfccs = extract_mfcc_for_timbre(audio_data, sample_rate)
    
    # Calculate pitch statistics (excluding NaN values)
    valid_pitch = pitch[~np.isnan(pitch)]
    pitch_stats = {}
    
    if len(valid_pitch) > 0:
        pitch_stats = {
            "mean": np.mean(valid_pitch),
            "median": np.median(valid_pitch),
            "std": np.std(valid_pitch),
            "min": np.min(valid_pitch),
            "max": np.max(valid_pitch),
            "range": np.max(valid_pitch) - np.min(valid_pitch)
        }
    
    # Calculate MFCC statistics
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds = np.std(mfccs, axis=1)
    
    mfcc_stats = {
        "means": mfcc_means.tolist(),
        "stds": mfcc_stds.tolist()
    }
    
    # Create plots
    pitch_fig = plot_pitch(
        pitch, 
        sample_rate, 
        title=f"{title_prefix} Pitch Estimation" if title_prefix else "Pitch Estimation",
        save_path=os.path.join(output_dir, "pitch.png") if output_dir else None
    )
    
    pitch_dist_fig = plot_pitch_distribution(
        pitch, 
        title=f"{title_prefix} Pitch Distribution" if title_prefix else "Pitch Distribution",
        save_path=os.path.join(output_dir, "pitch_distribution.png") if output_dir else None
    )
    
    mfcc_fig = plot_mfcc(
        mfccs, 
        sample_rate, 
        title=f"{title_prefix} MFCC Features" if title_prefix else "MFCC Features",
        save_path=os.path.join(output_dir, "mfcc.png") if output_dir else None
    )
    
    mfcc_stats_fig = plot_mfcc_statistics(
        mfccs, 
        title=f"{title_prefix} MFCC Statistics" if title_prefix else "MFCC Statistics",
        save_path=os.path.join(output_dir, "mfcc_statistics.png") if output_dir else None
    )
    
    # Return results
    results = {
        "pitch": pitch.tolist() if isinstance(pitch, np.ndarray) else pitch,
        "pitch_stats": pitch_stats,
        "mfcc_stats": mfcc_stats,
        "figures": {
            "pitch": pitch_fig,
            "pitch_distribution": pitch_dist_fig,
            "mfcc": mfcc_fig,
            "mfcc_statistics": mfcc_stats_fig
        }
    }
    
    return results

def batch_analyze_pitch_timbre(audio_data_dict: Dict[str, Tuple[np.ndarray, int]], 
                              output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Perform pitch and timbre analysis on a batch of audio files.
    
    Args:
        audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
        output_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary mapping file paths to analysis results.
    """
    logger.info(f"Batch analyzing pitch and timbre for {len(audio_data_dict)} files")
    results = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            # Create file-specific output directory if needed
            file_output_dir = None
            if output_dir:
                file_name = os.path.basename(file_path)
                file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0], "pitch_timbre")
                os.makedirs(file_output_dir, exist_ok=True)
            
            # Analyze this file
            file_results = analyze_pitch_timbre(
                audio_data, 
                sample_rate, 
                title_prefix=os.path.basename(file_path),
                output_dir=file_output_dir
            )
            
            results[file_path] = file_results
            logger.info(f"Completed pitch and timbre analysis for {file_path}")
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
    
    return results
