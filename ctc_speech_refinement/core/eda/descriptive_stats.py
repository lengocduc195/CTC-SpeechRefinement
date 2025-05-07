"""
Descriptive statistics analysis for audio data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_basic_stats(audio_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic descriptive statistics of audio data.

    Args:
        audio_data: Audio data as numpy array.

    Returns:
        Dictionary of basic statistics.
    """
    logger.info("Calculating basic statistics")

    stats = {
        "mean": np.mean(audio_data),
        "std": np.std(audio_data),
        "min": np.min(audio_data),
        "max": np.max(audio_data),
        "median": np.median(audio_data),
        "q1": np.percentile(audio_data, 25),
        "q3": np.percentile(audio_data, 75),
        "iqr": np.percentile(audio_data, 75) - np.percentile(audio_data, 25),
        "skewness": float(pd.Series(audio_data).skew()),
        "kurtosis": float(pd.Series(audio_data).kurtosis()),
        "rms": np.sqrt(np.mean(np.square(audio_data))),
        "crest_factor": np.max(np.abs(audio_data)) / np.sqrt(np.mean(np.square(audio_data))) if np.mean(np.square(audio_data)) > 0 else 0,
        "dynamic_range": np.max(audio_data) - np.min(audio_data),
        "zero_crossings": np.sum(librosa.zero_crossings(audio_data))
    }

    return stats

def plot_amplitude_distribution(audio_data: np.ndarray, title: str = "Amplitude Distribution",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the amplitude distribution of audio data.

    Args:
        audio_data: Audio data as numpy array.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting amplitude distribution")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(audio_data, bins=100, alpha=0.7, color='blue')

    # Plot normal distribution for comparison
    x = np.linspace(np.min(audio_data), np.max(audio_data), 100)
    mean = np.mean(audio_data)
    std = np.std(audio_data)
    normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    normal_dist = normal_dist * (len(audio_data) * (np.max(audio_data) - np.min(audio_data)) / 100)
    ax.plot(x, normal_dist, 'r--', linewidth=2)

    # Add vertical lines for key statistics
    ax.axvline(mean, color='g', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    ax.axvline(np.median(audio_data), color='m', linestyle='--', linewidth=2, label=f'Median: {np.median(audio_data):.4f}')
    ax.axvline(mean + std, color='y', linestyle='--', linewidth=1, label=f'Mean + Std: {mean + std:.4f}')
    ax.axvline(mean - std, color='y', linestyle='--', linewidth=1, label=f'Mean - Std: {mean - std:.4f}')

    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved amplitude distribution plot to {save_path}")

    return fig

def plot_stats_summary(stats: Dict[str, float], title: str = "Audio Statistics Summary",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot a summary of audio statistics.

    Args:
        stats: Dictionary of statistics.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting statistics summary")

    # Select key statistics to display
    key_stats = {
        "Mean": stats["mean"],
        "Std Dev": stats["std"],
        "Median": stats["median"],
        "Min": stats["min"],
        "Max": stats["max"],
        "RMS": stats["rms"],
        "Crest Factor": stats["crest_factor"],
        "Dynamic Range": stats["dynamic_range"],
        "Zero Crossings": stats["zero_crossings"]
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart
    bars = ax.bar(key_stats.keys(), key_stats.values(), color='skyblue')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 1000:
            value_text = f"{height:.2e}"
        else:
            value_text = f"{height:.4f}"
        ax.text(bar.get_x() + bar.get_width()/2., height,
                value_text, ha='center', va='bottom', rotation=0)

    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved statistics summary plot to {save_path}")

    return fig

def plot_dynamic_metrics(audio_data: np.ndarray, stats: Dict[str, float],
                        title: str = "Dynamic Range and Crest Factor",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot dynamic range and crest factor of audio data.

    Args:
        audio_data: Audio data as numpy array.
        stats: Dictionary of statistics.
        title: Title of the plot.
        save_path: Path to save the plot. If None, the plot is not saved.

    Returns:
        Matplotlib figure.
    """
    logger.info("Plotting dynamic metrics")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Dynamic Range
    ax1.hist(audio_data, bins=100, alpha=0.7, color='blue')
    ax1.axvline(stats["min"], color='r', linestyle='--', linewidth=1, label=f'Min: {stats["min"]:.4f}')
    ax1.axvline(stats["max"], color='g', linestyle='--', linewidth=1, label=f'Max: {stats["max"]:.4f}')
    ax1.set_xlabel('Amplitude')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Dynamic Range: {stats["dynamic_range"]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Crest Factor
    # Create a simple visualization of crest factor
    x = np.linspace(0, 1, 1000)
    y = np.sin(2 * np.pi * 5 * x)  # Simple sine wave
    peak = np.max(np.abs(y))
    rms = np.sqrt(np.mean(np.square(y)))
    crest = peak / rms

    ax2.plot(x, y, color='blue', alpha=0.7)
    ax2.axhline(peak, color='r', linestyle='--', linewidth=1, label=f'Peak: {peak:.4f}')
    ax2.axhline(rms, color='g', linestyle='--', linewidth=1, label=f'RMS: {rms:.4f}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Crest Factor Illustration (Audio CF: {stats["crest_factor"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved dynamic metrics plot to {save_path}")

    return fig

def analyze_descriptive_stats(audio_data: np.ndarray, sample_rate: int,
                             title_prefix: str = "", output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform comprehensive descriptive statistics analysis on audio data.

    Args:
        audio_data: Audio data as numpy array.
        sample_rate: Sample rate of the audio.
        title_prefix: Prefix for plot titles.
        output_dir: Directory to save plots. If None, plots are not saved.

    Returns:
        Dictionary containing statistics and figure objects.
    """
    logger.info("Performing descriptive statistics analysis")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Calculate basic statistics
    stats = calculate_basic_stats(audio_data)

    # Create plots
    amplitude_dist_fig = plot_amplitude_distribution(
        audio_data,
        title=f"{title_prefix} Amplitude Distribution" if title_prefix else "Amplitude Distribution",
        save_path=os.path.join(output_dir, "amplitude_distribution.png") if output_dir else None
    )

    stats_summary_fig = plot_stats_summary(
        stats,
        title=f"{title_prefix} Audio Statistics Summary" if title_prefix else "Audio Statistics Summary",
        save_path=os.path.join(output_dir, "stats_summary.png") if output_dir else None
    )

    dynamic_metrics_fig = plot_dynamic_metrics(
        audio_data,
        stats,
        title=f"{title_prefix} Dynamic Range and Crest Factor" if title_prefix else "Dynamic Range and Crest Factor",
        save_path=os.path.join(output_dir, "dynamic_metrics.png") if output_dir else None
    )

    # Add duration information
    duration = len(audio_data) / sample_rate
    stats["duration"] = duration
    stats["sample_rate"] = sample_rate
    stats["num_samples"] = len(audio_data)

    # Return results
    results = {
        "stats": stats,
        "figures": {
            "amplitude_distribution": amplitude_dist_fig,
            "stats_summary": stats_summary_fig,
            "dynamic_metrics": dynamic_metrics_fig
        }
    }

    return results

def batch_analyze_descriptive_stats(audio_data_dict: Dict[str, Tuple[np.ndarray, int]],
                                   output_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Perform descriptive statistics analysis on a batch of audio files.

    Args:
        audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
        output_dir: Directory to save plots. If None, plots are not saved.

    Returns:
        Dictionary mapping file paths to analysis results.
    """
    logger.info(f"Batch analyzing descriptive statistics for {len(audio_data_dict)} files")
    results = {}

    for file_path, (audio_data, sample_rate) in audio_data_dict.items():
        try:
            # Create file-specific output directory if needed
            file_output_dir = None
            if output_dir:
                file_name = os.path.basename(file_path)
                file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
                os.makedirs(file_output_dir, exist_ok=True)

            # Analyze this file
            file_results = analyze_descriptive_stats(
                audio_data,
                sample_rate,
                title_prefix=os.path.basename(file_path),
                output_dir=file_output_dir
            )

            results[file_path] = file_results
            logger.info(f"Completed descriptive statistics analysis for {file_path}")

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {str(e)}")

    return results
