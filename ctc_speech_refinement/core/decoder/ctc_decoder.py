"""
CTC decoder module for speech transcription.
"""

import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from pyctcdecode import build_ctcdecoder
import os

from config.config import (
    DECODER_TYPE, BEAM_WIDTH, ALPHA, BETA, USE_LM, LM_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTCDecoder:
    """
    CTC decoder for speech transcription.
    """
    
    def __init__(self, processor, decoder_type: str = DECODER_TYPE, 
                beam_width: int = BEAM_WIDTH, alpha: float = ALPHA, 
                beta: float = BETA, use_lm: bool = USE_LM, 
                lm_path: Optional[str] = LM_PATH):
        """
        Initialize the CTC decoder.
        
        Args:
            processor: Tokenizer processor from the acoustic model.
            decoder_type: Type of decoder to use. Options: "greedy", "beam_search".
            beam_width: Beam width for beam search decoding.
            alpha: Language model weight.
            beta: Word insertion bonus.
            use_lm: Whether to use a language model.
            lm_path: Path to language model file.
        """
        self.processor = processor
        self.decoder_type = decoder_type
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.use_lm = use_lm
        self.lm_path = lm_path
        
        logger.info(f"Initializing CTC decoder with type: {decoder_type}")
        
        # Get vocabulary from processor
        self.vocab = list(processor.tokenizer.get_vocab().keys())
        self.vocab = [v for v in self.vocab if not v.startswith("##")]
        
        # Special tokens
        self.blank_token = processor.tokenizer.pad_token
        self.blank_id = processor.tokenizer.pad_token_id
        
        # Initialize decoder
        if decoder_type == "beam_search" and use_lm and lm_path and os.path.exists(lm_path):
            logger.info(f"Using language model from {lm_path}")
            self.decoder = build_ctcdecoder(
                self.vocab,
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta
            )
        else:
            logger.info("Using decoder without language model")
            self.decoder = build_ctcdecoder(self.vocab)
    
    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding on logits.
        
        Args:
            logits: Logits from the acoustic model.
            
        Returns:
            Transcribed text.
        """
        logger.info("Performing greedy decoding")
        
        # Convert logits to numpy if they are torch tensors
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        # Get the most likely token at each timestep
        predictions = np.argmax(logits, axis=-1)
        
        # Convert token IDs to text
        if predictions.ndim > 1:
            predictions = predictions[0]  # Take the first batch item
            
        # Decode using the processor
        transcription = self.processor.decode(predictions)
        
        logger.info(f"Greedy decoding result: {transcription}")
        return transcription
    
    def beam_search_decode(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding on logits.
        
        Args:
            logits: Logits from the acoustic model.
            
        Returns:
            Transcribed text.
        """
        logger.info(f"Performing beam search decoding with beam width {self.beam_width}")
        
        # Convert logits to numpy if they are torch tensors
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        
        # Ensure logits are in the right shape
        if logits.ndim > 2:
            logits = logits[0]  # Take the first batch item
        
        # Convert logits to log probabilities
        log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1).numpy()
        
        # Decode using pyctcdecode
        transcription = self.decoder.decode(log_probs, beam_width=self.beam_width)
        
        logger.info(f"Beam search decoding result: {transcription}")
        return transcription
    
    def decode(self, logits: torch.Tensor) -> str:
        """
        Decode logits using the specified decoder type.
        
        Args:
            logits: Logits from the acoustic model.
            
        Returns:
            Transcribed text.
        """
        if self.decoder_type == "greedy":
            return self.greedy_decode(logits)
        elif self.decoder_type == "beam_search":
            return self.beam_search_decode(logits)
        else:
            raise ValueError(f"Unsupported decoder type: {self.decoder_type}")
    
    def batch_decode(self, logits_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        Decode a batch of logits.
        
        Args:
            logits_dict: Dictionary mapping file paths to logits.
            
        Returns:
            Dictionary mapping file paths to transcribed text.
        """
        logger.info(f"Batch decoding {len(logits_dict)} files")
        transcriptions = {}
        
        for file_path, logits in logits_dict.items():
            try:
                transcription = self.decode(logits)
                transcriptions[file_path] = transcription
                logger.info(f"Decoded {file_path}")
            except Exception as e:
                logger.error(f"Error decoding {file_path}: {str(e)}")
        
        return transcriptions
