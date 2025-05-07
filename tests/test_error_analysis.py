"""
Test script for the error analysis system.
"""

import os
import sys
import unittest
import tempfile
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda.error_analysis import ErrorAnalyzer
from src.eda.error_visualization import ErrorVisualizer
from src.eda.error_improvement import ErrorImprover

class TestErrorAnalysis(unittest.TestCase):
    """Test cases for the error analysis system."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample reference and hypothesis transcriptions
        self.reference_transcripts = {
            "file1.txt": "the quick brown fox jumps over the lazy dog",
            "file2.txt": "she sells sea shells by the sea shore",
            "file3.txt": "how much wood would a woodchuck chuck",
            "file4.txt": "peter piper picked a peck of pickled peppers",
            "file5.txt": "i scream you scream we all scream for ice cream"
        }
        
        self.hypothesis_transcripts = {
            "file1.txt": "the quick brown fox jumps over a lazy dog",  # Substitution: the -> a
            "file2.txt": "she sells sea shells the sea shore",         # Deletion: by
            "file3.txt": "how much wood would woodchuck chuck chuck",  # Deletion: a, Insertion: chuck
            "file4.txt": "peter piper picked peck of pickled papers",  # Deletion: a, Substitution: peppers -> papers
            "file5.txt": "i scream you scream we all for ice cream"    # Deletion: scream
        }
        
        # Create temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_error_analyzer(self):
        """Test the ErrorAnalyzer class."""
        # Initialize analyzer
        analyzer = ErrorAnalyzer(
            reference_transcripts=self.reference_transcripts,
            hypothesis_transcripts=self.hypothesis_transcripts
        )
        
        # Analyze errors
        analyzer.analyze_all_transcripts()
        
        # Check summary statistics
        self.assertTrue(hasattr(analyzer, 'summary_stats'))
        self.assertIn('avg_wer', analyzer.summary_stats)
        self.assertIn('avg_cer', analyzer.summary_stats)
        self.assertIn('total_files', analyzer.summary_stats)
        
        # Check that we have the expected number of files
        self.assertEqual(analyzer.summary_stats['total_files'], 5)
        
        # Generate report
        report = analyzer.generate_error_report(self.output_dir)
        
        # Check that report was generated
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'error_analysis_report.json')))
        
        # Check that report contains expected sections
        self.assertIn('summary', report)
        self.assertIn('detailed_errors', report)
        
        # Check that we have the expected error types
        self.assertIn('substitutions', report['detailed_errors'])
        self.assertIn('deletions', report['detailed_errors'])
        self.assertIn('insertions', report['detailed_errors'])
        
        # Check that we have the expected number of errors
        self.assertEqual(len(report['detailed_errors']['substitutions']), 2)  # the->a, peppers->papers
        self.assertEqual(len(report['detailed_errors']['deletions']), 3)      # by, a, scream
        self.assertEqual(len(report['detailed_errors']['insertions']), 1)     # chuck
    
    def test_error_visualizer(self):
        """Test the ErrorVisualizer class."""
        # First run the analyzer to get a report
        analyzer = ErrorAnalyzer(
            reference_transcripts=self.reference_transcripts,
            hypothesis_transcripts=self.hypothesis_transcripts
        )
        analyzer.analyze_all_transcripts()
        report = analyzer.generate_error_report(self.output_dir)
        
        # Initialize visualizer
        visualizer = ErrorVisualizer(report)
        
        # Create visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        visualizer.create_visualizations(vis_dir)
        
        # Check that visualization directory was created
        self.assertTrue(os.path.exists(vis_dir))
        
        # Check that expected visualization files were created
        expected_files = [
            'error_type_pie_chart.png',
            'wer_distribution.png'
        ]
        
        for file in expected_files:
            self.assertTrue(os.path.exists(os.path.join(vis_dir, file)))
    
    def test_error_improver(self):
        """Test the ErrorImprover class."""
        # First run the analyzer to get a report
        analyzer = ErrorAnalyzer(
            reference_transcripts=self.reference_transcripts,
            hypothesis_transcripts=self.hypothesis_transcripts
        )
        analyzer.analyze_all_transcripts()
        report = analyzer.generate_error_report(self.output_dir)
        
        # Initialize improver
        improver = ErrorImprover(report)
        
        # Analyze error patterns
        error_patterns = improver.analyze_error_patterns()
        
        # Check that error patterns were generated
        self.assertIn('substitution_patterns', error_patterns)
        self.assertIn('deletion_patterns', error_patterns)
        self.assertIn('insertion_patterns', error_patterns)
        self.assertIn('phonetic_confusions', error_patterns)
        
        # Generate and save improvement suggestions
        suggestions_path = improver.save_suggestions(self.output_dir)
        
        # Check that suggestions were saved
        self.assertTrue(os.path.exists(suggestions_path))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'improvement_suggestions.json')))
        
        # Load and check suggestions
        with open(os.path.join(self.output_dir, 'improvement_suggestions.json'), 'r') as f:
            suggestions = json.load(f)
        
        # Check that suggestions contain expected sections
        self.assertIn('acoustic_model_improvements', suggestions)
        self.assertIn('language_model_improvements', suggestions)
        self.assertIn('preprocessing_improvements', suggestions)
        self.assertIn('postprocessing_improvements', suggestions)
        self.assertIn('specific_word_fixes', suggestions)

if __name__ == '__main__':
    unittest.main()
