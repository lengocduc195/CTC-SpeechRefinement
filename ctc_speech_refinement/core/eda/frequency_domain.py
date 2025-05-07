"""
Frequency domain analysis for audio data.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path
from scipy import signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_fft(audio_data: np.ndarray, sample_rate: int, title: str = "FFT Magnitude Spectrum", 
            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the FFT magnitude spectrum of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting FFT magnitude spectrum")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute FFT
    n = len(audio_data)
    fft_data = np.abs(np.fft.rfft(audio_data))
    
    # Convert to dB
    fft_data_db = librosa.amplitude_to_db(fft_data, ref=np.max)
    
    # Create frequency array
    freq = np.fft.rfftfreq(n, d=1/sample_rate)
    
    # Plot FFT
    ax.plot(freq, fft_data_db, color='blue', alpha=0.7)
    
    # Find dominant frequency
    dominant_freq_idx = np.argmax(fft_data)
    dominant_freq = freq[dominant_freq_idx]
    
    # Mark dominant frequency
    ax.axvline(dominant_freq, color='r', linestyle='--', linewidth=1, 
              label=f'Dominant Freq: {dominant_freq:.1f} Hz')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to log scale for better visualization
    ax.set_xscale('log')
    ax.set_xlim([20, sample_rate/2])  # Focus on audible range
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved FFT plot to {save_path}")
    
    return fig

def plot_spectrogram(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                    hop_length: int = 512, title: str = "Spectrogram", 
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the spectrogram of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting spectrogram")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    
    # Plot spectrogram
    img = librosa.display.specshow(
        D, 
        sr=sample_rate, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='log',
        ax=ax
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved spectrogram plot to {save_path}")
    
    return fig

def plot_mel_spectrogram(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                        hop_length: int = 512, n_mels: int = 128, 
                        title: str = "Mel Spectrogram", 
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the mel spectrogram of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        n_mels: Number of mel bands.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting mel spectrogram")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    
    # Convert to dB
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Plot mel spectrogram
    img = librosa.display.specshow(
        S_dB, 
        sr=sample_rate, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='mel',
        ax=ax
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved mel spectrogram plot to {save_path}")
    
    return fig

def plot_chroma(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
               hop_length: int = 512, n_chroma: int = 12, 
               title: str = "Chromagram", 
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the chromagram of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        n_chroma: Number of chroma bins.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting chromagram")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute chromagram
    chroma = librosa.feature.chroma_stft(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_chroma=n_chroma
    )
    
    # Plot chromagram
    img = librosa.display.specshow(
        chroma, 
        sr=sample_rate, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='chroma',
        ax=ax
    )
    
    fig.colorbar(img, ax=ax)
    
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved chromagram plot to {save_path}")
    
    return fig

def plot_spectral_contrast(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                          hop_length: int = 512, n_bands: int = 6, 
                          title: str = "Spectral Contrast", 
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the spectral contrast of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        n_bands: Number of frequency bands.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting spectral contrast")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio_data, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_bands=n_bands
    )
    
    # Plot spectral contrast
    img = librosa.display.specshow(
        contrast, 
        sr=sample_rate, 
        hop_length=hop_length, 
        x_axis='time',
        ax=ax
    )
    
    fig.colorbar(img, ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('Frequency Bands')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved spectral contrast plot to {save_path}")
    
    return fig

def calculate_spectral_features(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                               hop_length: int = 512) -> Dict[str, float]:
    """
    Calculate spectral features of audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        
    Returns:
        Dictionary of spectral features.
    """
    logger.info("Calculating spectral features")
    
    # Compute spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio_data, n_fft=n_fft, hop_length=hop_length)[0]
    
    # Calculate statistics
    features = {
        "spectral_centroid_mean": np.mean(spectral_centroid),
        "spectral_centroid_std": np.std(spectral_centroid),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_std": np.std(spectral_bandwidth),
        "spectral_rolloff_mean": np.mean(spectral_rolloff),
        "spectral_rolloff_std": np.std(spectral_rolloff),
        "spectral_flatness_mean": np.mean(spectral_flatness),
        "spectral_flatness_std": np.std(spectral_flatness)
    }
    
    return features

def plot_spectral_features(audio_data: np.ndarray, sample_rate: int, n_fft: int = 2048, 
                          hop_length: int = 512, title: str = "Spectral Features", 
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot spectral features of audio data over time.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting spectral features")
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Compute spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio_data, n_fft=n_fft, hop_length=hop_length)[0]
    
    # Create time array
    frames = range(len(spectral_centroid))
    t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
    
    # Plot features
    axs[0].plot(t, spectral_centroid, color='blue')
    axs[0].set_ylabel('Spectral Centroid (Hz)')
    axs[0].set_title('Spectral Centroid')
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(t, spectral_bandwidth, color='green')
    axs[1].set_ylabel('Spectral Bandwidth (Hz)')
    axs[1].set_title('Spectral Bandwidth')
    axs[1].grid(True, alpha=0.3)
    
    axs[2].plot(t, spectral_rolloff, color='red')
    axs[2].set_ylabel('Spectral Rolloff (Hz)')
    axs[2].set_title('Spectral Rolloff')
    axs[2].grid(True, alpha=0.3)
    
    axs[3].plot(t, spectral_flatness, color='purple')
    axs[3].set_ylabel('Spectral Flatness')
    axs[3].set_title('Spectral Flatness')
    axs[3].grid(True, alpha=0.3)
    axs[3].set_xlabel('Time (s)')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved spectral features plot to {save_path}")
    
    return fig

def analyze_frequency_domain(audio_data: np.ndarray, sample_rate: int, 
                            title_prefix: str = "", output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive frequency domain analysis on audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        title_prefix: Prefix for plot titles.
        output_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary containing analysis results and figure objects.
    """
    logger.info("Performing frequency domain analysis")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate spectral features
    spectral_features = calculate_spectral_features(audio_data, sample_rate)
    
    # Create plots
    fft_fig = plot_fft(
        audio_data, 
        sample_rate, 
        title=f"{title_prefix} FFT Magnitude Spectrum" if title_prefix else "FFT Magnitude Spectrum",
        save_path=os.path.join(output_dir, "fft.png") if output_dir else None
    )
    
    spectrogram_fig = plot_spectrogram(
        audio_data, 
        sample_rate, 
        title=f"{title_prefix} Spectrogram" if title_prefix else "Spectrogram",
        save_path=os.path.join(output_dir, "spectrogram.png") if output_dir else None
    )
    
    mel_spectrogram_fig = plot_mel_spectrogram(
        audio_data, 
        sample_rate, 
        title=f"{title_prefix} Mel Spectrogram" if title_prefix else "Mel Spectrogram",
        save_path=os.path.join(output_dir, "mel_spectrogram.png") if output_dir else None
    )
    
    chroma_fig = plot_chroma(
        audio_data, 
        sample_rate, 
        title=f"{title_prefix} Chromagram" if title_prefix else "Chromagram",
        save_path=os.path.join(output_dir, "chroma.png") if output_dir else None
    )
    
    spectral_contrast_fig = plot_spectral_contrast(
        audio_data, 
        sample_rate, 
        title=f"{title_prefix} Spectral Contrast" if title_prefix else "Spectral Contrast",
        save_path=os.path.join(output_dir, "spectral_contrast.png") if output_dir else None
    )
    
    spectral_features_fig = plot_spectral_features(
        audio_data, 
        sample_rate, 
        title=f"{title_prefix} Spectral Features" if title_prefix else "Spectral Features",
        save_path=os.path.join(output_dir, "spectral_features.png") if output_dir else None
    )
    
    # Return results
    results = {
        "spectral_features": spectral_features,
        "figures": {
            "fft": fft_fig,
            "spectrogram": spectrogram_fig,
            "mel_spectrogram": mel_spectrogram_fig,
            "chroma": chroma_fig,
            "spectral_contrast": spectral_contrast_fig,
            "spectral_features": spectral_features_fig
        }
    }
    
    return results

def batch_analyze_frequency_domain(audio_data_dict: Dict[str, Tuple[np.ndarray, int]], 
                                  output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Perform frequency domain analysis on a batch of audio files.
    
    Args:
        audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
        output_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary mapping file paths to analysis results.
    """
    logger.info(f"Batch analyzing frequency domain for {len(audio_data_dict)} files")
    results = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            # Create file-specific output directory if needed
            file_output_dir = None
            if output_dir:
                file_name = os.path.basename(file_path)
                file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0], "frequency_domain")
                os.makedirs(file_output_dir, exist_ok=True)
            
            # Analyze this file
            file_results = analyze_frequency_domain(
                audio_data, 
                sample_rate, 
                title_prefix=os.path.basename(file_path),
                output_dir=file_output_dir
            )
            
            results[file_path] = file_results
            logger.info(f"Completed frequency domain analysis for {file_path}")
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
    
    return results
