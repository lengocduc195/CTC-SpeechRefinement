#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real Automatic Speech Recognition (ASR) for Vietnamese.

This script performs real speech recognition on audio files using pre-trained models
for Vietnamese. It supports both Wav2Vec2 and Whisper models.

Usage:
    python real_asr.py --input_dir data/audio --output_dir results/transcripts --model_type wav2vec2
"""

import os
import argparse
import logging
import time
import torch
import numpy as np
import soundfile as sf
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    WhisperForConditionalGeneration, 
    WhisperProcessor
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_asr.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Available pre-trained models for Vietnamese
AVAILABLE_MODELS = {
    "wav2vec2": {
        "default": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        "large": "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
    },
    "whisper": {
        "default": "vinai/PhoWhisper-large",
        "medium": "vinai/PhoWhisper-medium",
        "small": "vinai/PhoWhisper-small"
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Real ASR for Vietnamese")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/audio",
        help="Directory containing audio files to transcribe"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/transcripts",
        help="Directory to save transcriptions"
    )
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="wav2vec2",
        choices=["wav2vec2", "whisper"],
        help="Type of ASR model to use"
    )
    
    parser.add_argument(
        "--model_size", 
        type=str, 
        default="default",
        help="Size of the model to use (default, large, medium, small)"
    )
    
    parser.add_argument(
        "--custom_model_path", 
        type=str, 
        default=None,
        help="Path to a custom model (overrides model_type and model_size)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to use (cuda, cpu). If None, use CUDA if available."
    )
    
    parser.add_argument(
        "--reference_dir", 
        type=str, 
        default=None,
        help="Directory containing reference transcriptions for evaluation"
    )
    
    parser.add_argument(
        "--normalize_audio", 
        action="store_true",
        help="Whether to normalize audio"
    )
    
    return parser.parse_args()

def get_audio_files(directory: str) -> List[str]:
    """Get all audio files in a directory."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def load_audio(file_path: str, normalize: bool = False) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate."""
    try:
        # Load audio using soundfile
        audio_data, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Normalize if requested
        if normalize:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        # Return empty audio as fallback
        return np.zeros(16000), 16000

def resample_audio(audio_data: np.ndarray, orig_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sample_rate == target_sample_rate:
        return audio_data
    
    # Convert to torch tensor for resampling
    audio_tensor = torch.tensor(audio_data).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
    resampled_audio = resampler(audio_tensor).squeeze().numpy()
    
    return resampled_audio

class ASRModel:
    """Base class for ASR models."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize the ASR model."""
        self.model_path = model_path
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing ASR model {model_path} on device {self.device}")
        
        # To be implemented by subclasses
        self.model = None
        self.processor = None
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio data."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def batch_transcribe(self, audio_files: List[str], normalize: bool = False) -> Dict[str, str]:
        """Transcribe a batch of audio files."""
        transcriptions = {}
        
        for file_path in tqdm(audio_files, desc="Transcribing"):
            try:
                # Load and preprocess audio
                audio_data, sample_rate = load_audio(file_path, normalize)
                
                # Transcribe
                transcription = self.transcribe(audio_data, sample_rate)
                transcriptions[file_path] = transcription
                
                logger.info(f"Transcribed {file_path}: {transcription}")
            except Exception as e:
                logger.error(f"Error transcribing {file_path}: {str(e)}")
        
        return transcriptions

class Wav2Vec2Model(ASRModel):
    """Wav2Vec2 model for ASR."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize the Wav2Vec2 model."""
        super().__init__(model_path, device)
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
            logger.info(f"Successfully loaded Wav2Vec2 model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading Wav2Vec2 model {model_path}: {str(e)}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio data using Wav2Vec2."""
        # Resample to 16kHz if needed (Wav2Vec2 requires 16kHz)
        if sample_rate != 16000:
            audio_data = resample_audio(audio_data, sample_rate, 16000)
            sample_rate = 16000
        
        # Preprocess audio
        inputs = self.processor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        
        # Get logits
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription

class WhisperModel(ASRModel):
    """Whisper model for ASR."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """Initialize the Whisper model."""
        super().__init__(model_path, device)
        
        try:
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
            logger.info(f"Successfully loaded Whisper model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading Whisper model {model_path}: {str(e)}")
            raise
    
    def transcribe(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio data using Whisper."""
        # Resample to 16kHz if needed (Whisper requires 16kHz)
        if sample_rate != 16000:
            audio_data = resample_audio(audio_data, sample_rate, 16000)
            sample_rate = 16000
        
        # Preprocess audio
        inputs = self.processor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        
        # Generate tokens
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                language="vi",
                task="transcribe"
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription

def save_transcriptions(transcriptions: Dict[str, str], output_dir: str) -> Dict[str, str]:
    """Save transcriptions to files."""
    os.makedirs(output_dir, exist_ok=True)
    output_files = {}
    
    for audio_path, transcription in transcriptions.items():
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # Save transcription
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        output_files[audio_path] = output_path
    
    return output_files

def load_reference_transcriptions(reference_dir: str) -> Dict[str, str]:
    """Load reference transcriptions from files."""
    references = {}
    
    if not os.path.exists(reference_dir):
        logger.warning(f"Reference directory {reference_dir} does not exist")
        return references
    
    for root, _, files in os.walk(reference_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        references[base_name] = f.read().strip()
                except Exception as e:
                    logger.error(f"Error loading reference file {file_path}: {str(e)}")
    
    return references

def evaluate_transcriptions(references: Dict[str, str], transcriptions: Dict[str, str]) -> Dict[str, float]:
    """Evaluate transcriptions against references."""
    from jiwer import wer, cer
    
    results = {
        "wer": [],
        "cer": []
    }
    
    for audio_path, transcription in transcriptions.items():
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        if base_name in references:
            reference = references[base_name]
            
            # Calculate WER and CER
            try:
                results["wer"].append(wer(reference, transcription))
                results["cer"].append(cer(reference, transcription))
            except Exception as e:
                logger.error(f"Error calculating metrics for {base_name}: {str(e)}")
    
    # Calculate averages
    if results["wer"]:
        results["avg_wer"] = sum(results["wer"]) / len(results["wer"])
    else:
        results["avg_wer"] = 0.0
    
    if results["cer"]:
        results["avg_cer"] = sum(results["cer"]) / len(results["cer"])
    else:
        results["avg_cer"] = 0.0
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model path
    if args.custom_model_path:
        model_path = args.custom_model_path
    else:
        try:
            model_path = AVAILABLE_MODELS[args.model_type][args.model_size]
        except KeyError:
            logger.error(f"Invalid model type or size: {args.model_type}, {args.model_size}")
            logger.info(f"Available models: {AVAILABLE_MODELS}")
            return
    
    # Initialize model
    start_time = time.time()
    logger.info(f"Initializing {args.model_type} model: {model_path}")
    
    try:
        if args.model_type == "wav2vec2":
            model = Wav2Vec2Model(model_path, device=args.device)
        elif args.model_type == "whisper":
            model = WhisperModel(model_path, device=args.device)
        else:
            logger.error(f"Unsupported model type: {args.model_type}")
            return
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return
    
    logger.info(f"Model initialization completed in {time.time() - start_time:.2f} seconds")
    
    # Get audio files
    audio_files = get_audio_files(args.input_dir)
    if not audio_files:
        logger.error(f"No audio files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files in {args.input_dir}")
    
    # Transcribe audio files
    start_time = time.time()
    logger.info("Transcribing audio files...")
    transcriptions = model.batch_transcribe(audio_files, normalize=args.normalize_audio)
    logger.info(f"Transcription completed in {time.time() - start_time:.2f} seconds")
    
    # Save transcriptions
    logger.info(f"Saving transcriptions to {args.output_dir}")
    output_files = save_transcriptions(transcriptions, args.output_dir)
    
    # Evaluate if reference directory is provided
    if args.reference_dir:
        logger.info(f"Loading reference transcriptions from {args.reference_dir}")
        references = load_reference_transcriptions(args.reference_dir)
        
        if references:
            logger.info("Evaluating transcriptions against references")
            results = evaluate_transcriptions(references, transcriptions)
            
            logger.info(f"Average WER: {results['avg_wer']:.4f}")
            logger.info(f"Average CER: {results['avg_cer']:.4f}")
        else:
            logger.warning("No reference transcriptions found for evaluation")
    
    logger.info("ASR process completed successfully")

if __name__ == "__main__":
    main()
