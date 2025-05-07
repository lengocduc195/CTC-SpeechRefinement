"""
Evaluation utilities for speech transcription.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Union
from jiwer import wer, cer
import os
from pathlib import Path

from config.config import RESULTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.
        
    Returns:
        Word Error Rate.
    """
    try:
        return wer(reference, hypothesis)
    except Exception as e:
        logger.error(f"Error calculating WER: {str(e)}")
        return 1.0  # Return worst possible score on error

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.
    
    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.
        
    Returns:
        Character Error Rate.
    """
    try:
        return cer(reference, hypothesis)
    except Exception as e:
        logger.error(f"Error calculating CER: {str(e)}")
        return 1.0  # Return worst possible score on error

def evaluate_transcriptions(references: Dict[str, str], hypotheses: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate transcriptions against references.
    
    Args:
        references: Dictionary mapping file paths to reference transcriptions.
        hypotheses: Dictionary mapping file paths to hypothesis transcriptions.
        
    Returns:
        Dictionary mapping file paths to dictionaries of metrics.
    """
    logger.info(f"Evaluating {len(hypotheses)} transcriptions")
    results = {}
    
    for file_path, hypothesis in hypotheses.items():
        if file_path in references:
            reference = references[file_path]
            
            wer_score = calculate_wer(reference, hypothesis)
            cer_score = calculate_cer(reference, hypothesis)
            
            results[file_path] = {
                "wer": wer_score,
                "cer": cer_score
            }
            
            logger.info(f"Evaluation for {file_path}: WER={wer_score:.4f}, CER={cer_score:.4f}")
        else:
            logger.warning(f"No reference found for {file_path}")
    
    return results

def calculate_average_metrics(evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Calculate average metrics across all files.
    
    Args:
        evaluation_results: Dictionary mapping file paths to dictionaries of metrics.
        
    Returns:
        Dictionary of average metrics.
    """
    logger.info("Calculating average metrics")
    
    if not evaluation_results:
        logger.warning("No evaluation results to average")
        return {"avg_wer": 1.0, "avg_cer": 1.0}
    
    wer_scores = [results["wer"] for results in evaluation_results.values() if "wer" in results]
    cer_scores = [results["cer"] for results in evaluation_results.values() if "cer" in results]
    
    avg_wer = np.mean(wer_scores) if wer_scores else 1.0
    avg_cer = np.mean(cer_scores) if cer_scores else 1.0
    
    logger.info(f"Average metrics: WER={avg_wer:.4f}, CER={avg_cer:.4f}")
    
    return {
        "avg_wer": avg_wer,
        "avg_cer": avg_cer
    }

def save_evaluation_results(evaluation_results: Dict[str, Dict[str, float]], 
                           output_dir: Optional[str] = None, 
                           filename: str = "evaluation_results.csv") -> str:
    """
    Save evaluation results to a CSV file.
    
    Args:
        evaluation_results: Dictionary mapping file paths to dictionaries of metrics.
        output_dir: Directory to save the results. If None, use the default results directory.
        filename: Name of the output file.
        
    Returns:
        Path to the saved file.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    logger.info(f"Saving evaluation results to {output_path}")
    
    # Convert to DataFrame
    data = []
    for file_path, metrics in evaluation_results.items():
        row = {"file_path": file_path}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add average metrics
    avg_metrics = calculate_average_metrics(evaluation_results)
    avg_row = {"file_path": "AVERAGE"}
    avg_row.update(avg_metrics)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    return output_path

def plot_metrics(evaluation_results: Dict[str, Dict[str, float]], 
                output_dir: Optional[str] = None, 
                filename: str = "metrics_plot.png") -> str:
    """
    Plot metrics for each file.
    
    Args:
        evaluation_results: Dictionary mapping file paths to dictionaries of metrics.
        output_dir: Directory to save the plot. If None, use the default results directory.
        filename: Name of the output file.
        
    Returns:
        Path to the saved plot.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    logger.info(f"Creating metrics plot at {output_path}")
    
    # Extract file names and metrics
    file_names = [os.path.basename(file_path) for file_path in evaluation_results.keys()]
    wer_scores = [results["wer"] for results in evaluation_results.values() if "wer" in results]
    cer_scores = [results["cer"] for results in evaluation_results.values() if "cer" in results]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(file_names))
    width = 0.35
    
    ax.bar(x - width/2, wer_scores, width, label='WER')
    ax.bar(x + width/2, cer_scores, width, label='CER')
    
    ax.set_xlabel('Files')
    ax.set_ylabel('Error Rate')
    ax.set_title('Word and Character Error Rates by File')
    ax.set_xticks(x)
    ax.set_xticklabels(file_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path
