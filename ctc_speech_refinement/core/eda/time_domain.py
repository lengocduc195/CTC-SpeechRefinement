"""
Time domain analysis for audio data.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path
from scipy.signal import find_peaks, hilbert

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_waveform(audio_data: np.ndarray, sample_rate: int, title: str = "Audio Waveform",
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the waveform of audio data.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting audio waveform")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create time array
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, len(audio_data))

    # Plot waveform
    ax.plot(time, audio_data, color='blue', alpha=0.7)

    # Add horizontal lines for key statistics
    mean = np.mean(audio_data)
    std = np.std(audio_data)
    ax.axhline(mean, color='r', linestyle='--', linewidth=1, label=f'Mean: {mean:.4f}')
    ax.axhline(mean + std, color='g', linestyle='--', linewidth=1, label=f'Mean + Std: {mean + std:.4f}')
    ax.axhline(mean - std, color='g', linestyle='--', linewidth=1, label=f'Mean - Std: {mean - std:.4f}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved waveform plot to {save_path}")

    return fig

def plot_envelope(audio_data: np.ndarray, sample_rate: int, title: str = "Audio Envelope",
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the envelope of audio data using Hilbert transform and RMS methods.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting audio envelope")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Create time array
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, len(audio_data))

    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(audio_data)
    hilbert_envelope = np.abs(analytic_signal)

    # Plot waveform and Hilbert envelope
    ax1.plot(time, audio_data, color='blue', alpha=0.5, label='Waveform')
    ax1.plot(time, hilbert_envelope, color='red', linewidth=2, label='Hilbert Envelope')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f"{title} - Hilbert Transform Method")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Calculate envelope using RMS method
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.times_like(rms, sr=sample_rate, hop_length=hop_length)

    # Calculate envelope using peak detection
    frame_size = int(0.03 * sample_rate)  # 30ms frames
    hop_size = int(0.015 * sample_rate)   # 15ms hop

    peak_envelope = []
    for i in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[i:i+frame_size]
        peak_envelope.append(np.max(np.abs(frame)))

    peak_times = np.linspace(0, duration, len(peak_envelope))

    # Plot RMS and peak envelopes
    ax2.plot(time, audio_data, color='blue', alpha=0.5, label='Waveform')
    ax2.plot(rms_times, rms, color='green', linewidth=2, label='RMS Envelope')
    ax2.plot(peak_times, peak_envelope, color='purple', linewidth=2, label='Peak Envelope')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f"{title} - RMS and Peak Methods")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved envelope plot to {save_path}")

    return fig

def plot_energy(audio_data: np.ndarray, sample_rate: int, frame_length: int = 2048,
               hop_length: int = 512, title: str = "Audio Energy",
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the energy of audio data over time.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        frame_length: Length of each frame in samples.
        hop_length: Number of samples between frames.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting audio energy")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate energy
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]

    # Create time array for energy
    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)

    # Plot energy
    ax.plot(t, energy, color='green', linewidth=2)

    # Add horizontal line for mean energy
    mean_energy = np.mean(energy)
    ax.axhline(mean_energy, color='r', linestyle='--', linewidth=1, label=f'Mean Energy: {mean_energy:.4f}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (RMS)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved energy plot to {save_path}")

    return fig

def plot_zero_crossings(audio_data: np.ndarray, sample_rate: int, frame_length: int = 2048,
                       hop_length: int = 512, title: str = "Zero Crossing Rate",
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the zero crossing rate of audio data over time.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        frame_length: Length of each frame in samples.
        hop_length: Number of samples between frames.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting zero crossing rate")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_length, hop_length=hop_length)[0]

    # Create time array for ZCR
    frames = range(len(zcr))
    t = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)

    # Plot ZCR
    ax.plot(t, zcr, color='purple', linewidth=2)

    # Add horizontal line for mean ZCR
    mean_zcr = np.mean(zcr)
    ax.axhline(mean_zcr, color='r', linestyle='--', linewidth=1, label=f'Mean ZCR: {mean_zcr:.4f}')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Zero Crossing Rate')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved zero crossing rate plot to {save_path}")

    return fig

def detect_silence(audio_data: np.ndarray, sample_rate: int, threshold: float = 0.01,
                  min_silence_duration: float = 0.1) -> List[Tuple[float, float]]:
    """
    Detect silent regions in audio data.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        threshold: Energy threshold below which audio is considered silent.
        min_silence_duration: Minimum duration of silence in seconds.

    Returns:
        List of (start_time, end_time) tuples for silent regions.
    """
    logger.info("Detecting silent regions")

    # Calculate energy
    energy = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]

    # Find frames below threshold
    silent_frames = np.where(energy < threshold)[0]

    # Convert to time
    silent_times = librosa.frames_to_time(silent_frames, sr=sample_rate, hop_length=512)

    # Group consecutive silent frames
    silent_regions = []
    if len(silent_times) > 0:
        # Initialize with the first silent time
        start_time = silent_times[0]
        prev_time = silent_times[0]

        for i in range(1, len(silent_times)):
            current_time = silent_times[i]

            # If there's a gap, end the current region and start a new one
            if current_time - prev_time > 0.1:  # 0.1s gap threshold
                end_time = prev_time

                # Only add if the region is long enough
                if end_time - start_time >= min_silence_duration:
                    silent_regions.append((start_time, end_time))

                start_time = current_time

            prev_time = current_time

        # Add the last region
        if len(silent_times) > 0:
            end_time = silent_times[-1]
            if end_time - start_time >= min_silence_duration:
                silent_regions.append((start_time, end_time))

    logger.info(f"Detected {len(silent_regions)} silent regions")
    return silent_regions

def plot_silence_detection(audio_data: np.ndarray, sample_rate: int, silent_regions: List[Tuple[float, float]],
                          title: str = "Silence Detection", save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot audio waveform with highlighted silent regions.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        silent_regions: List of (start_time, end_time) tuples for silent regions.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting silence detection")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create time array
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, len(audio_data))

    # Plot waveform
    ax.plot(time, audio_data, color='blue', alpha=0.7)

    # Highlight silent regions
    for start_time, end_time in silent_regions:
        ax.axvspan(start_time, end_time, color='red', alpha=0.3)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add legend
    import matplotlib.patches as mpatches
    silent_patch = mpatches.Patch(color='red', alpha=0.3, label='Silent Regions')
    ax.legend(handles=[silent_patch], loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved silence detection plot to {save_path}")

    return fig

def analyze_time_domain(audio_data: np.ndarray, sample_rate: int,
                       title_prefix: str = "", output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive time domain analysis on audio data.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        title_prefix: Prefix for plot titles.
        output_dir: Directory to save plots. If None, plots are not saved.

    Returns:
        Dictionary containing analysis results and figure objects.
    """
    logger.info("Performing time domain analysis")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create plots
    waveform_fig = plot_waveform(
        audio_data,
        sample_rate,
        title=f"{title_prefix} Audio Waveform" if title_prefix else "Audio Waveform",
        save_path=os.path.join(output_dir, "waveform.png") if output_dir else None
    )

    envelope_fig = plot_envelope(
        audio_data,
        sample_rate,
        title=f"{title_prefix} Audio Envelope" if title_prefix else "Audio Envelope",
        save_path=os.path.join(output_dir, "envelope.png") if output_dir else None
    )

    energy_fig = plot_energy(
        audio_data,
        sample_rate,
        title=f"{title_prefix} Audio Energy" if title_prefix else "Audio Energy",
        save_path=os.path.join(output_dir, "energy.png") if output_dir else None
    )

    zcr_fig = plot_zero_crossings(
        audio_data,
        sample_rate,
        title=f"{title_prefix} Zero Crossing Rate" if title_prefix else "Zero Crossing Rate",
        save_path=os.path.join(output_dir, "zero_crossings.png") if output_dir else None
    )

    # Detect silent regions
    silent_regions = detect_silence(audio_data, sample_rate)

    silence_fig = plot_silence_detection(
        audio_data,
        sample_rate,
        silent_regions,
        title=f"{title_prefix} Silence Detection" if title_prefix else "Silence Detection",
        save_path=os.path.join(output_dir, "silence_detection.png") if output_dir else None
    )

    # Return results
    results = {
        "silent_regions": silent_regions,
        "figures": {
            "waveform": waveform_fig,
            "envelope": envelope_fig,
            "energy": energy_fig,
            "zero_crossings": zcr_fig,
            "silence_detection": silence_fig
        }
    }

    return results

def batch_analyze_time_domain(audio_data_dict: Dict[str, Tuple[np.ndarray, int]],
                             output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Perform time domain analysis on a batch of audio files.

    Args:
        audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
        output_dir: Directory to save plots. If None, plots are not saved.

    Returns:
        Dictionary mapping file paths to analysis results.
    """
    logger.info(f"Batch analyzing time domain for {len(audio_data_dict)} files")
    results = {}

    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            # Create file-specific output directory if needed
            file_output_dir = None
            if output_dir:
                file_name = os.path.basename(file_path)
                file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0], "time_domain")
                os.makedirs(file_output_dir, exist_ok=True)

            # Analyze this file
            file_results = analyze_time_domain(
                audio_data,
                sample_rate,
                title_prefix=os.path.basename(file_path),
                output_dir=file_output_dir
            )

            results[file_path] = file_results
            logger.info(f"Completed time domain analysis for {file_path}")

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")

    return results
