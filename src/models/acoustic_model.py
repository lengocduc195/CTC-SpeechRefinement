"""
Acoustic model module for CTC Speech Transcription.
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path

from config.config import PRETRAINED_MODEL_NAME, MODELS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AcousticModel:
    """
    Wrapper class for acoustic models used in CTC speech transcription.
    """
    
    def __init__(self, model_name: str = PRETRAINED_MODEL_NAME, device: Optional[str] = None):
        """
        Initialize the acoustic model.
        
        Args:
            model_name: Name of the pretrained model or path to a local model.
            device: Device to use for inference. If None, use CUDA if available, else CPU.
        """
        self.model_name = model_name
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing acoustic model {model_name} on device {self.device}")
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            logger.info(f"Successfully loaded model and processor from {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio data for the model.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Preprocessed audio data as torch tensor.
        """
        logger.info(f"Preprocessing audio data with shape {audio_data.shape}")
        inputs = self.processor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(self.device)
        return input_values
    
    def get_logits(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """
        Get logits from the model for the given audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Logits as torch tensor.
        """
        logger.info("Getting logits from model")
        input_values = self.preprocess_audio(audio_data, sample_rate)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            logits = outputs.logits
        
        logger.info(f"Logits shape: {logits.shape}")
        return logits
    
    def get_probabilities(self, audio_data: np.ndarray, sample_rate: int) -> torch.Tensor:
        """
        Get probabilities from the model for the given audio data.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Probabilities as torch tensor.
        """
        logger.info("Getting probabilities from model")
        logits = self.get_logits(audio_data, sample_rate)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        logger.info(f"Probabilities shape: {probs.shape}")
        return probs
    
    def decode_greedy(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """
        Perform greedy decoding on the model output.
        
        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.
            
        Returns:
            Transcribed text.
        """
        logger.info("Performing greedy decoding")
        input_values = self.preprocess_audio(audio_data, sample_rate)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        logger.info(f"Transcription: {transcription[0]}")
        return transcription[0]
    
    def save_model(self, output_dir: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model. If None, use the default models directory.
            
        Returns:
            Path to the saved model.
        """
        if output_dir is None:
            output_dir = os.path.join(MODELS_DIR, os.path.basename(self.model_name))
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model to {output_dir}")
        
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        return output_dir
    
    @classmethod
    def load_model(cls, model_path: str, device: Optional[str] = None) -> 'AcousticModel':
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model.
            device: Device to use for inference. If None, use CUDA if available, else CPU.
            
        Returns:
            Loaded AcousticModel instance.
        """
        logger.info(f"Loading model from {model_path}")
        return cls(model_name=model_path, device=device)
    
    def batch_process(self, audio_data_dict: Dict[str, Tuple[np.ndarray, int]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of audio data and get logits.
        
        Args:
            audio_data_dict: Dictionary mapping file paths to tuples of (audio_data, sample_rate).
            
        Returns:
            Dictionary mapping file paths to logits.
        """
        logger.info(f"Batch processing {len(audio_data_dict)} audio files")
        logits_dict = {}
        
        for file_path, (audio_data, sample_rate) in audio_data_dict.items():
            try:
                logits = self.get_logits(audio_data, sample_rate)
                logits_dict[file_path] = logits
                logger.info(f"Processed {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        
        return logits_dict
