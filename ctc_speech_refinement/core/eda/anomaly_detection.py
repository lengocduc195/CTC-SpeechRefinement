"""
Anomaly detection for audio data.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_amplitude_anomalies(audio_data: np.ndarray, threshold: float = 3.0) -> List[Tuple[int, int]]:
    """
    Detect amplitude anomalies in audio data using z-score.
    
    Args:
        audio_data: Audio data as numpy array.
        threshold: Z-score threshold for anomaly detection.
        
    Returns:
        List of (start_index, end_index) tuples for anomalous regions.
    """
    logger.info(f"Detecting amplitude anomalies with threshold {threshold}")
    
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(audio_data))
    
    # Find anomalous samples
    anomalies = np.where(z_scores > threshold)[0]
    
    # Group consecutive anomalies
    anomalous_regions = []
    if len(anomalies) > 0:
        # Initialize with the first anomaly
        start_idx = anomalies[0]
        prev_idx = anomalies[0]
        
        for i in range(1, len(anomalies)):
            current_idx = anomalies[i]
            
            # If there's a gap, end the current region and start a new one
            if current_idx - prev_idx > 1:
                end_idx = prev_idx
                anomalous_regions.append((start_idx, end_idx))
                start_idx = current_idx
            
            prev_idx = current_idx
        
        # Add the last region
        if len(anomalies) > 0:
            end_idx = anomalies[-1]
            anomalous_regions.append((start_idx, end_idx))
    
    logger.info(f"Detected {len(anomalous_regions)} amplitude anomalous regions")
    return anomalous_regions

def detect_spectral_anomalies(audio_data: np.ndarray, sample_rate: int, 
                             n_fft: int = 2048, hop_length: int = 512, 
                             contamination: float = 0.05) -> List[int]:
    """
    Detect spectral anomalies in audio data using Isolation Forest.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        n_fft: FFT window size.
        hop_length: Number of samples between frames.
        contamination: Expected proportion of anomalies.
        
    Returns:
        List of frame indices for anomalous frames.
    """
    logger.info(f"Detecting spectral anomalies with contamination {contamination}")
    
    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    
    spectral_flatness = librosa.feature.spectral_flatness(
        y=audio_data, n_fft=n_fft, hop_length=hop_length)[0]
    
    # Combine features
    features = np.vstack([
        spectral_centroid, 
        spectral_bandwidth, 
        spectral_rolloff, 
        spectral_flatness
    ]).T
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(features_scaled)
    
    # Find anomalous frames (-1 indicates an anomaly)
    anomalous_frames = np.where(predictions == -1)[0]
    
    logger.info(f"Detected {len(anomalous_frames)} spectral anomalous frames")
    return anomalous_frames.tolist()

def plot_amplitude_anomalies(audio_data: np.ndarray, sample_rate: int, 
                            anomalous_regions: List[Tuple[int, int]], 
                            title: str = "Amplitude Anomalies", 
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot audio waveform with highlighted amplitude anomalies.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        anomalous_regions: List of (start_index, end_index) tuples for anomalous regions.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting amplitude anomalies")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create time array
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, len(audio_data))
    
    # Plot waveform
    ax.plot(time, audio_data, color='blue', alpha=0.7)
    
    # Highlight anomalous regions
    for start_idx, end_idx in anomalous_regions:
        start_time = start_idx / sample_rate
        end_time = end_idx / sample_rate
        ax.axvspan(start_time, end_time, color='red', alpha=0.3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    import matplotlib.patches as mpatches
    anomaly_patch = mpatches.Patch(color='red', alpha=0.3, label='Anomalies')
    ax.legend(handles=[anomaly_patch], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved amplitude anomalies plot to {save_path}")
    
    return fig

def plot_spectral_anomalies(audio_data: np.ndarray, sample_rate: int, 
                           anomalous_frames: List[int], hop_length: int = 512, 
                           title: str = "Spectral Anomalies", 
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot spectrogram with highlighted spectral anomalies.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        anomalous_frames: List of frame indices for anomalous frames.
        hop_length: Number of samples between frames.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.
        
    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting spectral anomalies")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_data, hop_length=hop_length)),
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
    
    # Highlight anomalous frames
    for frame in anomalous_frames:
        time = librosa.frames_to_time(frame, sr=sample_rate, hop_length=hop_length)
        ax.axvline(time, color='red', alpha=0.5, linewidth=1)
    
    ax.set_title(title)
    
    # Add legend
    import matplotlib.lines as mlines
    anomaly_line = mlines.Line2D([], [], color='red', alpha=0.5, linewidth=1, label='Anomalies')
    ax.legend(handles=[anomaly_line], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved spectral anomalies plot to {save_path}")
    
    return fig

def detect_outliers_in_features(features: np.ndarray, contamination: float = 0.05) -> List[int]:
    """
    Detect outliers in feature vectors using Isolation Forest.
    
    Args:
        features: Feature vectors as numpy array of shape (n_samples, n_features).
        contamination: Expected proportion of outliers.
        
    Returns:
        List of indices for outlier samples.
    """
    logger.info(f"Detecting outliers in features with contamination {contamination}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(features_scaled)
    
    # Find outlier samples (-1 indicates an outlier)
    outliers = np.where(predictions == -1)[0]
    
    logger.info(f"Detected {len(outliers)} outliers")
    return outliers.tolist()

def analyze_anomalies(audio_data: np.ndarray, sample_rate: int, 
                     amplitude_threshold: float = 3.0, spectral_contamination: float = 0.05, 
                     title_prefix: str = "", output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive anomaly detection on audio data.
    
    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        amplitude_threshold: Z-score threshold for amplitude anomaly detection.
        spectral_contamination: Expected proportion of spectral anomalies.
        title_prefix: Prefix for plot titles.
        output_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary containing analysis results and figure objects.
    """
    logger.info("Performing anomaly detection")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Detect amplitude anomalies
    amplitude_anomalies = detect_amplitude_anomalies(audio_data, threshold=amplitude_threshold)
    
    # Detect spectral anomalies
    spectral_anomalies = detect_spectral_anomalies(
        audio_data, 
        sample_rate, 
        contamination=spectral_contamination
    )
    
    # Create plots
    amplitude_anomalies_fig = plot_amplitude_anomalies(
        audio_data, 
        sample_rate, 
        amplitude_anomalies,
        title=f"{title_prefix} Amplitude Anomalies" if title_prefix else "Amplitude Anomalies",
        save_path=os.path.join(output_dir, "amplitude_anomalies.png") if output_dir else None
    )
    
    spectral_anomalies_fig = plot_spectral_anomalies(
        audio_data, 
        sample_rate, 
        spectral_anomalies,
        title=f"{title_prefix} Spectral Anomalies" if title_prefix else "Spectral Anomalies",
        save_path=os.path.join(output_dir, "spectral_anomalies.png") if output_dir else None
    )
    
    # Return results
    results = {
        "amplitude_anomalies": [list(region) for region in amplitude_anomalies],
        "spectral_anomalies": spectral_anomalies,
        "figures": {
            "amplitude_anomalies": amplitude_anomalies_fig,
            "spectral_anomalies": spectral_anomalies_fig
        }
    }
    
    return results

def batch_analyze_anomalies(audio_data_dict: Dict[str, Tuple[np.ndarray, int]], 
                           amplitude_threshold: float = 3.0, 
                           spectral_contamination: float = 0.05, 
                           output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Perform anomaly detection on a batch of audio files.
    
    Args:
        audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
        amplitude_threshold: Z-score threshold for amplitude anomaly detection.
        spectral_contamination: Expected proportion of spectral anomalies.
        output_dir: Directory to save plots. If None, plots are not saved.
        
    Returns:
        Dictionary mapping file paths to analysis results.
    """
    logger.info(f"Batch analyzing anomalies for {len(audio_data_dict)} files")
    results = {}
    
    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            # Create file-specific output directory if needed
            file_output_dir = None
            if output_dir:
                file_name = os.path.basename(file_path)
                file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0], "anomalies")
                os.makedirs(file_output_dir, exist_ok=True)
            
            # Analyze this file
            file_results = analyze_anomalies(
                audio_data, 
                sample_rate, 
                amplitude_threshold=amplitude_threshold,
                spectral_contamination=spectral_contamination,
                title_prefix=os.path.basename(file_path),
                output_dir=file_output_dir
            )
            
            results[file_path] = file_results
            logger.info(f"Completed anomaly detection for {file_path}")
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")
    
    return results
