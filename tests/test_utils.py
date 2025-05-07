"""
Unit tests for utility functions.
"""

import unittest
import os
import tempfile
import json
import numpy as np
from pathlib import Path

from src.utils.evaluation import (
    calculate_wer, calculate_cer, evaluate_transcriptions,
    calculate_average_metrics, save_evaluation_results
)
from src.utils.file_utils import (
    save_transcription, save_transcriptions, load_transcription,
    save_json, load_json
)

class TestEvaluation(unittest.TestCase):
    """Test cases for evaluation functions."""
    
    def test_calculate_wer(self):
        """Test Word Error Rate calculation."""
        reference = "the quick brown fox jumps over the lazy dog"
        hypothesis = "the quick brown fox jumps over the lazy"
        
        # Expected WER: 1 word error (deletion) out of 9 words = 1/9 = 0.111...
        expected_wer = 1 / 9
        wer = calculate_wer(reference, hypothesis)
        
        self.assertAlmostEqual(wer, expected_wer, places=3)
    
    def test_calculate_cer(self):
        """Test Character Error Rate calculation."""
        reference = "hello world"
        hypothesis = "hello worl"
        
        # Expected CER: 1 character error (deletion) out of 11 characters = 1/11 = 0.0909...
        expected_cer = 1 / 11
        cer = calculate_cer(reference, hypothesis)
        
        self.assertAlmostEqual(cer, expected_cer, places=3)
    
    def test_evaluate_transcriptions(self):
        """Test transcription evaluation."""
        references = {
            "file1.wav": "the quick brown fox",
            "file2.wav": "jumps over the lazy dog"
        }
        
        hypotheses = {
            "file1.wav": "the quick brown",
            "file2.wav": "jumps over the lazy dog"
        }
        
        results = evaluate_transcriptions(references, hypotheses)
        
        # Check that we have results for both files
        self.assertEqual(len(results), 2)
        
        # Check that the keys in the results dictionary match the input file paths
        self.assertSetEqual(set(results.keys()), set(references.keys()))
        
        # Check that each result has WER and CER metrics
        for metrics in results.values():
            self.assertIn("wer", metrics)
            self.assertIn("cer", metrics)
        
        # Check that the WER for file1 is non-zero (there's an error)
        self.assertGreater(results["file1.wav"]["wer"], 0)
        
        # Check that the WER for file2 is zero (perfect match)
        self.assertEqual(results["file2.wav"]["wer"], 0)
    
    def test_calculate_average_metrics(self):
        """Test average metrics calculation."""
        evaluation_results = {
            "file1.wav": {"wer": 0.25, "cer": 0.1},
            "file2.wav": {"wer": 0.0, "cer": 0.0},
            "file3.wav": {"wer": 0.5, "cer": 0.2}
        }
        
        avg_metrics = calculate_average_metrics(evaluation_results)
        
        # Check that we have average WER and CER
        self.assertIn("avg_wer", avg_metrics)
        self.assertIn("avg_cer", avg_metrics)
        
        # Check that the average WER is correct: (0.25 + 0.0 + 0.5) / 3 = 0.25
        self.assertAlmostEqual(avg_metrics["avg_wer"], 0.25, places=6)
        
        # Check that the average CER is correct: (0.1 + 0.0 + 0.2) / 3 = 0.1
        self.assertAlmostEqual(avg_metrics["avg_cer"], 0.1, places=6)
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            evaluation_results = {
                "file1.wav": {"wer": 0.25, "cer": 0.1},
                "file2.wav": {"wer": 0.0, "cer": 0.0}
            }
            
            output_path = save_evaluation_results(evaluation_results, temp_dir)
            
            # Check that the output file exists
            self.assertTrue(os.path.exists(output_path))
            
            # Check that the file is not empty
            self.assertGreater(os.path.getsize(output_path), 0)

class TestFileUtils(unittest.TestCase):
    """Test cases for file utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_save_load_transcription(self):
        """Test saving and loading a transcription."""
        transcription = "This is a test transcription."
        file_path = "audio.wav"
        output_dir = self.temp_dir.name
        
        # Save the transcription
        output_file = save_transcription(transcription, file_path, output_dir)
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Load the transcription
        loaded_transcription = load_transcription(output_file)
        
        # Check that the loaded transcription matches the original
        self.assertEqual(loaded_transcription, transcription)
    
    def test_save_transcriptions(self):
        """Test saving multiple transcriptions."""
        transcriptions = {
            "file1.wav": "This is the first transcription.",
            "file2.wav": "This is the second transcription."
        }
        output_dir = self.temp_dir.name
        
        # Save the transcriptions
        output_files = save_transcriptions(transcriptions, output_dir)
        
        # Check that we have output files for both transcriptions
        self.assertEqual(len(output_files), 2)
        
        # Check that all output files exist
        for output_file in output_files.values():
            self.assertTrue(os.path.exists(output_file))
    
    def test_save_load_json(self):
        """Test saving and loading JSON data."""
        data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
            "key4": {"nested": "value"}
        }
        file_path = os.path.join(self.temp_dir.name, "test.json")
        
        # Save the JSON data
        save_json(data, file_path)
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(file_path))
        
        # Load the JSON data
        loaded_data = load_json(file_path)
        
        # Check that the loaded data matches the original
        self.assertEqual(loaded_data, data)

if __name__ == "__main__":
    unittest.main()
