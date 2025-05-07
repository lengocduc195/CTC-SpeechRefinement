"""
File utility functions for speech transcription.
"""

import os
import glob
import json
import logging
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

from ctc_speech_refinement.config.config import TRANSCRIPTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_audio_files(directory: str, extensions: List[str] = [".wav", ".mp3", ".flac"]) -> List[str]:
    """
    Get all audio files in a directory with the specified extensions.
    
    Args:
        directory: Directory to search for audio files.
        extensions: List of audio file extensions to include.
        
    Returns:
        List of paths to audio files.
    """
    logger.info(f"Searching for audio files in {directory}")
    audio_files = []
    
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        audio_files.extend(glob.glob(pattern))
    
    logger.info(f"Found {len(audio_files)} audio files")
    return sorted(audio_files)

def save_transcription(transcription: str, file_path: str, 
                      output_dir: Optional[str] = None) -> str:
    """
    Save a transcription to a text file.
    
    Args:
        transcription: Transcription text.
        file_path: Path to the original audio file.
        output_dir: Directory to save the transcription. If None, use the default transcripts directory.
        
    Returns:
        Path to the saved transcription file.
    """
    if output_dir is None:
        output_dir = TRANSCRIPTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base name of the audio file and change the extension to .txt
    base_name = os.path.basename(file_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_dir, f"{base_name_no_ext}.txt")
    
    logger.info(f"Saving transcription to {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    return output_file

def save_transcriptions(transcriptions: Dict[str, str], 
                       output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Save multiple transcriptions to text files.
    
    Args:
        transcriptions: Dictionary mapping file paths to transcriptions.
        output_dir: Directory to save the transcriptions. If None, use the default transcripts directory.
        
    Returns:
        Dictionary mapping original file paths to transcription file paths.
    """
    logger.info(f"Saving {len(transcriptions)} transcriptions")
    output_files = {}
    
    for file_path, transcription in transcriptions.items():
        output_file = save_transcription(transcription, file_path, output_dir)
        output_files[file_path] = output_file
    
    return output_files

def load_transcription(file_path: str) -> str:
    """
    Load a transcription from a text file.
    
    Args:
        file_path: Path to the transcription file.
        
    Returns:
        Transcription text.
    """
    logger.info(f"Loading transcription from {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        transcription = f.read().strip()
    
    return transcription

def load_transcriptions(directory: str) -> Dict[str, str]:
    """
    Load all transcriptions from a directory.
    
    Args:
        directory: Directory containing transcription files.
        
    Returns:
        Dictionary mapping file paths to transcriptions.
    """
    logger.info(f"Loading transcriptions from {directory}")
    transcriptions = {}
    
    for file_path in glob.glob(os.path.join(directory, "*.txt")):
        try:
            transcription = load_transcription(file_path)
            
            # Convert the transcription file path to the corresponding audio file path
            base_name = os.path.basename(file_path)
            base_name_no_ext = os.path.splitext(base_name)[0]
            
            # This assumes the audio files have the same base name but different extension
            # You might need to adjust this logic based on your file naming convention
            audio_file = base_name_no_ext  # Just store the base name without extension
            
            transcriptions[audio_file] = transcription
            
        except Exception as e:
            logger.error(f"Error loading transcription from {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(transcriptions)} transcriptions")
    return transcriptions

def save_json(data: Any, file_path: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save.
        file_path: Path to the output file.
    """
    logger.info(f"Saving JSON data to {file_path}")
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        Loaded data.
    """
    logger.info(f"Loading JSON data from {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data
