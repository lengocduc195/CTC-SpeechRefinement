"""
Main module for audio exploratory data analysis (EDA).
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import pandas as pd

from src.preprocessing.audio import preprocess_audio, batch_preprocess
from src.eda.preprocessing import preprocess_for_eda, batch_preprocess_for_eda
from src.utils.file_utils import get_audio_files
from src.eda.descriptive_stats import analyze_descriptive_stats, batch_analyze_descriptive_stats
from src.eda.time_domain import analyze_time_domain, batch_analyze_time_domain
from src.eda.frequency_domain import analyze_frequency_domain, batch_analyze_frequency_domain
from src.eda.pitch_timbre import analyze_pitch_timbre, batch_analyze_pitch_timbre
from src.eda.anomaly_detection import analyze_anomalies, batch_analyze_anomalies

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_audio_file(file_path: str, output_dir: Optional[str] = None,
                      normalize: bool = True, remove_silence: bool = False,
                      trim_start: float = 0.0, trim_end: Optional[float] = None,
                      fade_in: float = 0.0, fade_out: float = 0.0,
                      filter_type: Optional[str] = None, cutoff_freq: Union[float, Tuple[float, float]] = 1000.0,
                      add_noise_level: Optional[float] = None) -> Dict[str, Any]:
    """
    Perform comprehensive EDA on a single audio file.

    Args:
        file_path: Path to the audio file.
        output_dir: Directory to save analysis results and plots. If None, plots are not saved.
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
        Dictionary containing analysis results.
    """
    logger.info(f"Analyzing audio file: {file_path}")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load audio
    audio_data, sample_rate = preprocess_audio(file_path, normalize=False, remove_silence_flag=False)

    # Apply additional preprocessing for EDA
    audio_data = preprocess_for_eda(
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

    # Create file-specific output directory
    file_output_dir = None
    if output_dir:
        file_name = os.path.basename(file_path)
        file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
        os.makedirs(file_output_dir, exist_ok=True)

    # Perform various analyses
    descriptive_stats_results = analyze_descriptive_stats(
        audio_data,
        sample_rate,
        title_prefix=os.path.basename(file_path),
        output_dir=os.path.join(file_output_dir, "descriptive_stats") if file_output_dir else None
    )

    time_domain_results = analyze_time_domain(
        audio_data,
        sample_rate,
        title_prefix=os.path.basename(file_path),
        output_dir=os.path.join(file_output_dir, "time_domain") if file_output_dir else None
    )

    frequency_domain_results = analyze_frequency_domain(
        audio_data,
        sample_rate,
        title_prefix=os.path.basename(file_path),
        output_dir=os.path.join(file_output_dir, "frequency_domain") if file_output_dir else None
    )

    pitch_timbre_results = analyze_pitch_timbre(
        audio_data,
        sample_rate,
        title_prefix=os.path.basename(file_path),
        output_dir=os.path.join(file_output_dir, "pitch_timbre") if file_output_dir else None
    )

    anomaly_results = analyze_anomalies(
        audio_data,
        sample_rate,
        title_prefix=os.path.basename(file_path),
        output_dir=os.path.join(file_output_dir, "anomalies") if file_output_dir else None
    )

    # Combine results
    results = {
        "file_path": file_path,
        "sample_rate": sample_rate,
        "duration": len(audio_data) / sample_rate,
        "num_samples": len(audio_data),
        "descriptive_stats": descriptive_stats_results["stats"],
        "time_domain": {
            "silent_regions": time_domain_results["silent_regions"]
        },
        "frequency_domain": {
            "spectral_features": frequency_domain_results["spectral_features"]
        },
        "pitch_timbre": {
            "pitch_stats": pitch_timbre_results["pitch_stats"],
            "mfcc_stats": pitch_timbre_results["mfcc_stats"]
        },
        "anomalies": {
            "amplitude_anomalies": anomaly_results["amplitude_anomalies"],
            "spectral_anomalies": anomaly_results["spectral_anomalies"]
        }
    }

    # Save results to JSON if output directory is specified
    if file_output_dir:
        results_path = os.path.join(file_output_dir, "analysis_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = results.copy()
            json.dump(json_results, f, indent=2)
        logger.info(f"Saved analysis results to {results_path}")

    logger.info(f"Completed analysis for {file_path}")
    return results

def batch_analyze_audio_files(file_paths: List[str], output_dir: Optional[str] = None,
                             normalize: bool = True, remove_silence: bool = False,
                             trim_start: float = 0.0, trim_end: Optional[float] = None,
                             fade_in: float = 0.0, fade_out: float = 0.0,
                             filter_type: Optional[str] = None, cutoff_freq: Union[float, Tuple[float, float]] = 1000.0,
                             add_noise_level: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """
    Perform comprehensive EDA on multiple audio files.

    Args:
        file_paths: List of paths to audio files.
        output_dir: Directory to save analysis results and plots. If None, plots are not saved.
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
        Dictionary mapping file paths to analysis results.
    """
    logger.info(f"Batch analyzing {len(file_paths)} audio files")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load all audio files
    audio_data_dict = batch_preprocess(file_paths, normalize=False, remove_silence_flag=False)

    # Apply additional preprocessing for EDA
    audio_data_dict = batch_preprocess_for_eda(
        audio_data_dict,
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

    # Perform various analyses
    descriptive_stats_results = batch_analyze_descriptive_stats(audio_data_dict, output_dir)
    time_domain_results = batch_analyze_time_domain(audio_data_dict, output_dir)
    frequency_domain_results = batch_analyze_frequency_domain(audio_data_dict, output_dir)
    pitch_timbre_results = batch_analyze_pitch_timbre(audio_data_dict, output_dir)
    anomaly_results = batch_analyze_anomalies(audio_data_dict, output_dir=output_dir)

    # Combine results
    results = {}
    for file_path in file_paths:
        if file_path in audio_data_dict:
            audio_data, sample_rate = audio_data_dict[file_path]

            results[file_path] = {
                "file_path": file_path,
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "num_samples": len(audio_data)
            }

            # Add descriptive stats
            if file_path in descriptive_stats_results:
                results[file_path]["descriptive_stats"] = descriptive_stats_results[file_path]["stats"]

            # Add time domain results
            if file_path in time_domain_results:
                results[file_path]["time_domain"] = {
                    "silent_regions": time_domain_results[file_path]["silent_regions"]
                }

            # Add frequency domain results
            if file_path in frequency_domain_results:
                results[file_path]["frequency_domain"] = {
                    "spectral_features": frequency_domain_results[file_path]["spectral_features"]
                }

            # Add pitch and timbre results
            if file_path in pitch_timbre_results:
                results[file_path]["pitch_timbre"] = {
                    "pitch_stats": pitch_timbre_results[file_path]["pitch_stats"],
                    "mfcc_stats": pitch_timbre_results[file_path]["mfcc_stats"]
                }

            # Add anomaly results
            if file_path in anomaly_results:
                results[file_path]["anomalies"] = {
                    "amplitude_anomalies": anomaly_results[file_path]["amplitude_anomalies"],
                    "spectral_anomalies": anomaly_results[file_path]["spectral_anomalies"]
                }

            # Save individual results to JSON
            if output_dir:
                file_name = os.path.basename(file_path)
                file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
                os.makedirs(file_output_dir, exist_ok=True)

                results_path = os.path.join(file_output_dir, "analysis_results.json")
                with open(results_path, 'w') as f:
                    json.dump(results[file_path], f, indent=2)
                logger.info(f"Saved analysis results to {results_path}")

    # Generate summary report
    if output_dir and results:
        generate_summary_report(results, output_dir)

    logger.info(f"Completed batch analysis for {len(results)} files")
    return results

def generate_summary_report(results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
    """
    Generate a summary report of the analysis results.

    Args:
        results: Dictionary mapping file paths to analysis results.
        output_dir: Directory to save the report.

    Returns:
        Path to the generated report.
    """
    logger.info("Generating summary report")

    # Create a DataFrame for the summary
    summary_data = []

    for file_path, result in results.items():
        file_name = os.path.basename(file_path)

        # Extract key metrics
        row = {
            "File Name": file_name,
            "Duration (s)": result.get("duration", 0),
            "Sample Rate (Hz)": result.get("sample_rate", 0),
            "Mean Amplitude": result.get("descriptive_stats", {}).get("mean", 0),
            "RMS Amplitude": result.get("descriptive_stats", {}).get("rms", 0),
            "Zero Crossings": result.get("descriptive_stats", {}).get("zero_crossings", 0),
            "Mean Pitch (Hz)": result.get("pitch_timbre", {}).get("pitch_stats", {}).get("mean", 0),
            "Spectral Centroid (Hz)": result.get("frequency_domain", {}).get("spectral_features", {}).get("spectral_centroid_mean", 0),
            "Spectral Bandwidth (Hz)": result.get("frequency_domain", {}).get("spectral_features", {}).get("spectral_bandwidth_mean", 0),
            "Spectral Flatness": result.get("frequency_domain", {}).get("spectral_features", {}).get("spectral_flatness_mean", 0),
            "Amplitude Anomalies": len(result.get("anomalies", {}).get("amplitude_anomalies", [])),
            "Spectral Anomalies": len(result.get("anomalies", {}).get("spectral_anomalies", []))
        }

        summary_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, "summary_report.csv")
    df.to_csv(csv_path, index=False)

    # Generate HTML report
    html_path = os.path.join(output_dir, "summary_report.html")

    with open(html_path, 'w') as f:
        f.write("<html>\n<head>\n")
        f.write("<title>Audio Analysis Summary Report</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("h1 { color: #2c3e50; }\n")
        f.write("table { border-collapse: collapse; width: 100%; }\n")
        f.write("th, td { text-align: left; padding: 8px; }\n")
        f.write("th { background-color: #2c3e50; color: white; }\n")
        f.write("tr:nth-child(even) { background-color: #f2f2f2; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>Audio Analysis Summary Report</h1>\n")
        f.write("<p>Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>\n")
        f.write("<h2>Summary Statistics</h2>\n")
        f.write(df.to_html(index=False))
        f.write("\n</body>\n</html>")

    logger.info(f"Saved summary report to {html_path}")
    return html_path

def analyze_directory(directory: str, output_dir: Optional[str] = None,
                     normalize: bool = True, remove_silence: bool = False,
                     trim_start: float = 0.0, trim_end: Optional[float] = None,
                     fade_in: float = 0.0, fade_out: float = 0.0,
                     filter_type: Optional[str] = None, cutoff_freq: Union[float, Tuple[float, float]] = 1000.0,
                     add_noise_level: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """
    Perform comprehensive EDA on all audio files in a directory.

    Args:
        directory: Directory containing audio files.
        output_dir: Directory to save analysis results and plots. If None, plots are not saved.
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
        Dictionary mapping file paths to analysis results.
    """
    logger.info(f"Analyzing audio files in directory: {directory}")

    # Get all audio files in the directory
    file_paths = get_audio_files(directory)

    if not file_paths:
        logger.warning(f"No audio files found in {directory}")
        return {}

    # Analyze all files
    return batch_analyze_audio_files(
        file_paths,
        output_dir,
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
