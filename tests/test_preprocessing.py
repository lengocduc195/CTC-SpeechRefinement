"""
Unit tests for the audio preprocessing module.
"""

import unittest
import numpy as np
import os
import tempfile
from pathlib import Path
import soundfile as sf

from src.preprocessing.audio import (
    normalize_audio, remove_silence, preprocess_audio, batch_preprocess
)

class TestPreprocessing(unittest.TestCase):
    """Test cases for audio preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a synthetic audio signal (1 second of 440Hz sine wave at 16kHz)
        self.sample_rate = 16000
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Add some silence
        silence = np.zeros(int(0.2 * self.sample_rate))
        self.audio_with_silence = np.concatenate([silence, self.audio_data, silence])
        
        # Save the audio to a temporary file
        self.audio_path = os.path.join(self.temp_dir.name, "test_audio.wav")
        sf.write(self.audio_path, self.audio_data, self.sample_rate)
        
        # Save the audio with silence to a temporary file
        self.audio_with_silence_path = os.path.join(self.temp_dir.name, "test_audio_with_silence.wav")
        sf.write(self.audio_with_silence_path, self.audio_with_silence, self.sample_rate)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        normalized_audio = normalize_audio(self.audio_data)
        
        # Check that the mean is close to 0
        self.assertAlmostEqual(np.mean(normalized_audio), 0, places=6)
        
        # Check that the standard deviation is close to 1
        self.assertAlmostEqual(np.std(normalized_audio), 1, places=6)
    
    def test_normalize_audio_constant(self):
        """Test audio normalization with constant signal."""
        constant_audio = np.ones(1000)
        normalized_audio = normalize_audio(constant_audio)
        
        # Check that the mean is 0
        self.assertAlmostEqual(np.mean(normalized_audio), 0, places=6)
        
        # Check that the values are all the same
        self.assertTrue(np.allclose(normalized_audio, np.zeros_like(normalized_audio)))
    
    def test_remove_silence(self):
        """Test silence removal."""
        audio_without_silence = remove_silence(self.audio_with_silence, self.sample_rate)
        
        # Check that the output is shorter than the input
        self.assertLess(len(audio_without_silence), len(self.audio_with_silence))
        
        # Check that the output is not too short (should preserve most of the actual audio)
        self.assertGreater(len(audio_without_silence), 0.7 * len(self.audio_data))
    
    def test_preprocess_audio(self):
        """Test the complete preprocessing pipeline."""
        processed_audio, sr = preprocess_audio(
            self.audio_with_silence_path, 
            normalize=True, 
            remove_silence_flag=True
        )
        
        # Check that the sample rate is correct
        self.assertEqual(sr, self.sample_rate)
        
        # Check that the output is shorter than the input (due to silence removal)
        self.assertLess(len(processed_audio), len(self.audio_with_silence))
        
        # Check that the mean is close to 0 (due to normalization)
        self.assertAlmostEqual(np.mean(processed_audio), 0, places=6)
        
        # Check that the standard deviation is close to 1 (due to normalization)
        self.assertAlmostEqual(np.std(processed_audio), 1, places=6)
    
    def test_batch_preprocess(self):
        """Test batch preprocessing."""
        file_paths = [self.audio_path, self.audio_with_silence_path]
        results = batch_preprocess(
            file_paths, 
            normalize=True, 
            remove_silence_flag=True
        )
        
        # Check that we have results for both files
        self.assertEqual(len(results), 2)
        
        # Check that the keys in the results dictionary match the input file paths
        self.assertSetEqual(set(results.keys()), set(file_paths))
        
        # Check that each result is a tuple of (audio_data, sample_rate)
        for audio_data, sample_rate in results.values():
            self.assertIsInstance(audio_data, np.ndarray)
            self.assertEqual(sample_rate, self.sample_rate)
            
            # Check that the mean is close to 0 (due to normalization)
            self.assertAlmostEqual(np.mean(audio_data), 0, places=6)
            
            # Check that the standard deviation is close to 1 (due to normalization)
            self.assertAlmostEqual(np.std(audio_data), 1, places=6)

if __name__ == "__main__":
    unittest.main()
