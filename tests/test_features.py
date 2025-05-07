"""
Unit tests for the feature extraction module.
"""

import unittest
import numpy as np
import os
import tempfile
from pathlib import Path

from src.features.extraction import (
    extract_mfcc, extract_mel_spectrogram, extract_spectrogram,
    extract_features, normalize_features, pad_features, batch_extract_features
)

class TestFeatureExtraction(unittest.TestCase):
    """Test cases for feature extraction functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a synthetic audio signal (1 second of 440Hz sine wave at 16kHz)
        self.sample_rate = 16000
        self.duration = 1.0
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        self.audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Create a dictionary of audio data for batch processing
        self.audio_data_dict = {
            "file1.wav": (self.audio_data, self.sample_rate),
            "file2.wav": (self.audio_data, self.sample_rate)
        }
    
    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        n_mfcc = 13
        mfccs = extract_mfcc(self.audio_data, self.sample_rate, n_mfcc=n_mfcc)
        
        # Check the shape of the output
        self.assertEqual(mfccs.shape[0], n_mfcc)
        self.assertGreater(mfccs.shape[1], 0)
        
        # Check that the output is a numpy array
        self.assertIsInstance(mfccs, np.ndarray)
    
    def test_extract_mel_spectrogram(self):
        """Test Mel spectrogram extraction."""
        n_mels = 128
        mel_spec = extract_mel_spectrogram(self.audio_data, self.sample_rate, n_mels=n_mels)
        
        # Check the shape of the output
        self.assertEqual(mel_spec.shape[0], n_mels)
        self.assertGreater(mel_spec.shape[1], 0)
        
        # Check that the output is a numpy array
        self.assertIsInstance(mel_spec, np.ndarray)
    
    def test_extract_spectrogram(self):
        """Test spectrogram extraction."""
        spec = extract_spectrogram(self.audio_data)
        
        # Check that the output has two dimensions
        self.assertEqual(len(spec.shape), 2)
        self.assertGreater(spec.shape[0], 0)
        self.assertGreater(spec.shape[1], 0)
        
        # Check that the output is a numpy array
        self.assertIsInstance(spec, np.ndarray)
    
    def test_extract_features_mfcc(self):
        """Test feature extraction with MFCC."""
        features = extract_features(self.audio_data, self.sample_rate, feature_type="mfcc")
        
        # Check that the output has two dimensions
        self.assertEqual(len(features.shape), 2)
        self.assertGreater(features.shape[0], 0)
        self.assertGreater(features.shape[1], 0)
        
        # Check that the output is a numpy array
        self.assertIsInstance(features, np.ndarray)
    
    def test_extract_features_mel_spectrogram(self):
        """Test feature extraction with Mel spectrogram."""
        features = extract_features(self.audio_data, self.sample_rate, feature_type="mel_spectrogram")
        
        # Check that the output has two dimensions
        self.assertEqual(len(features.shape), 2)
        self.assertGreater(features.shape[0], 0)
        self.assertGreater(features.shape[1], 0)
        
        # Check that the output is a numpy array
        self.assertIsInstance(features, np.ndarray)
    
    def test_extract_features_spectrogram(self):
        """Test feature extraction with spectrogram."""
        features = extract_features(self.audio_data, self.sample_rate, feature_type="spectrogram")
        
        # Check that the output has two dimensions
        self.assertEqual(len(features.shape), 2)
        self.assertGreater(features.shape[0], 0)
        self.assertGreater(features.shape[1], 0)
        
        # Check that the output is a numpy array
        self.assertIsInstance(features, np.ndarray)
    
    def test_extract_features_invalid_type(self):
        """Test feature extraction with invalid feature type."""
        with self.assertRaises(ValueError):
            extract_features(self.audio_data, self.sample_rate, feature_type="invalid_type")
    
    def test_normalize_features(self):
        """Test feature normalization."""
        features = np.random.randn(13, 100)
        normalized_features = normalize_features(features)
        
        # Check that the mean along the specified axis is close to 0
        self.assertTrue(np.allclose(np.mean(normalized_features, axis=0), 0, atol=1e-6))
        
        # Check that the standard deviation along the specified axis is close to 1
        self.assertTrue(np.allclose(np.std(normalized_features, axis=0), 1, atol=1e-6))
    
    def test_pad_features(self):
        """Test feature padding."""
        features = np.random.randn(13, 100)
        max_length = 150
        padded_features = pad_features(features, max_length)
        
        # Check that the padded features have the correct shape
        self.assertEqual(padded_features.shape, (13, max_length))
        
        # Check that the original features are preserved
        self.assertTrue(np.allclose(padded_features[:, :100], features))
        
        # Check that the padding is zeros
        self.assertTrue(np.allclose(padded_features[:, 100:], 0))
    
    def test_pad_features_truncate(self):
        """Test feature padding with truncation."""
        features = np.random.randn(13, 100)
        max_length = 50
        padded_features = pad_features(features, max_length)
        
        # Check that the padded features have the correct shape
        self.assertEqual(padded_features.shape, (13, max_length))
        
        # Check that the features are truncated correctly
        self.assertTrue(np.allclose(padded_features, features[:, :max_length]))
    
    def test_batch_extract_features(self):
        """Test batch feature extraction."""
        features_dict = batch_extract_features(self.audio_data_dict, feature_type="mfcc", normalize=True)
        
        # Check that we have features for both files
        self.assertEqual(len(features_dict), 2)
        
        # Check that the keys in the features dictionary match the input file paths
        self.assertSetEqual(set(features_dict.keys()), set(self.audio_data_dict.keys()))
        
        # Check that each feature is a numpy array with the correct shape
        for features in features_dict.values():
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(len(features.shape), 2)
            self.assertGreater(features.shape[0], 0)
            self.assertGreater(features.shape[1], 0)

if __name__ == "__main__":
    unittest.main()
