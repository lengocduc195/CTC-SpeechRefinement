"""
Error analysis module for speech recognition results.

This module provides functions to analyze and categorize errors in speech recognition results
by comparing them with reference transcriptions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import re
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import Levenshtein
from jiwer import wer, cer
import nltk
from nltk.metrics import edit_distance
from nltk.util import ngrams

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ErrorAnalyzer:
    """
    Analyzer for speech recognition errors.
    
    This class provides methods to analyze and categorize errors in speech recognition results
    by comparing them with reference transcriptions.
    """
    
    def __init__(self, reference_transcripts: Dict[str, str], 
                hypothesis_transcripts: Dict[str, str],
                language: str = "english"):
        """
        Initialize the ErrorAnalyzer.
        
        Args:
            reference_transcripts: Dictionary mapping file paths to reference transcriptions.
            hypothesis_transcripts: Dictionary mapping file paths to hypothesis (recognized) transcriptions.
            language: Language of the transcriptions for tokenization.
        """
        self.reference_transcripts = reference_transcripts
        self.hypothesis_transcripts = hypothesis_transcripts
        self.language = language
        
        # Ensure we have matching keys
        self.common_keys = set(reference_transcripts.keys()).intersection(set(hypothesis_transcripts.keys()))
        if len(self.common_keys) < len(reference_transcripts) or len(self.common_keys) < len(hypothesis_transcripts):
            logger.warning(f"Some transcripts don't have matching pairs. Using {len(self.common_keys)} common files.")
        
        # Initialize error statistics
        self.error_stats = {
            "wer": {},
            "cer": {},
            "substitutions": {},
            "deletions": {},
            "insertions": {},
            "pronunciation_errors": {},
            "dictionary_errors": {},
            "suffix_errors": {},
            "nonsense_errors": {},
            "boundary_errors": {},
            "repetition_errors": {}
        }
        
        # Initialize detailed error analysis
        self.detailed_errors = {
            "substitutions": defaultdict(int),  # word -> replacement count
            "deletions": defaultdict(int),      # word -> deletion count
            "insertions": defaultdict(int),     # word -> insertion count
            "context_errors": []                # List of error contexts
        }
        
        logger.info(f"Initialized ErrorAnalyzer with {len(self.common_keys)} transcript pairs")
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of tokens (words).
        """
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s\-]', '', text.lower())
        # Split on whitespace
        return text.split()
    
    def align_sequences(self, reference: List[str], hypothesis: List[str]) -> List[Tuple[str, str]]:
        """
        Align reference and hypothesis sequences using Levenshtein distance.
        
        Args:
            reference: List of reference tokens.
            hypothesis: List of hypothesis tokens.
            
        Returns:
            List of aligned token pairs (reference_token, hypothesis_token).
            None in either position indicates insertion or deletion.
        """
        # Initialize the alignment matrix
        m, n = len(reference), len(hypothesis)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the matrix
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if reference[i-1] == hypothesis[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + 1,  # substitution
                                  dp[i-1][j] + 1,     # deletion
                                  dp[i][j-1] + 1)     # insertion
        
        # Backtrack to find the alignment
        alignment = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and reference[i-1] == hypothesis[j-1]:
                alignment.append((reference[i-1], hypothesis[j-1]))
                i -= 1
                j -= 1
            elif j > 0 and (i == 0 or dp[i][j-1] + 1 == dp[i][j]):
                alignment.append((None, hypothesis[j-1]))  # Insertion
                j -= 1
            elif i > 0 and (j == 0 or dp[i-1][j] + 1 == dp[i][j]):
                alignment.append((reference[i-1], None))  # Deletion
                i -= 1
            else:
                alignment.append((reference[i-1], hypothesis[j-1]))  # Substitution
                i -= 1
                j -= 1
        
        # Reverse the alignment to get the correct order
        return alignment[::-1]
    
    def analyze_errors(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Analyze errors between reference and hypothesis transcriptions.
        
        Args:
            reference: Reference transcription.
            hypothesis: Hypothesis (recognized) transcription.
            
        Returns:
            Dictionary containing error analysis results.
        """
        # Tokenize the texts
        ref_tokens = self.tokenize_text(reference)
        hyp_tokens = self.tokenize_text(hypothesis)
        
        # Calculate WER and CER
        error_stats = {
            "wer": wer(reference, hypothesis),
            "cer": cer(reference, hypothesis)
        }
        
        # Align the sequences
        alignment = self.align_sequences(ref_tokens, hyp_tokens)
        
        # Count error types
        substitutions = []
        deletions = []
        insertions = []
        
        for ref_token, hyp_token in alignment:
            if ref_token is not None and hyp_token is not None and ref_token != hyp_token:
                substitutions.append((ref_token, hyp_token))
            elif ref_token is not None and hyp_token is None:
                deletions.append(ref_token)
            elif ref_token is None and hyp_token is not None:
                insertions.append(hyp_token)
        
        error_stats["substitutions"] = substitutions
        error_stats["deletions"] = deletions
        error_stats["insertions"] = insertions
        
        # Analyze specific error patterns
        error_stats.update(self.analyze_specific_errors(ref_tokens, hyp_tokens, alignment))
        
        return error_stats
    
    def analyze_specific_errors(self, ref_tokens: List[str], hyp_tokens: List[str], 
                               alignment: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze specific error patterns.
        
        Args:
            ref_tokens: Reference tokens.
            hyp_tokens: Hypothesis tokens.
            alignment: Aligned token pairs.
            
        Returns:
            Dictionary containing specific error analysis results.
        """
        specific_errors = {
            "pronunciation_errors": [],
            "dictionary_errors": [],
            "suffix_errors": [],
            "nonsense_errors": [],
            "boundary_errors": [],
            "repetition_errors": []
        }
        
        # Analyze pronunciation errors (similar sounding words)
        for ref_token, hyp_token in alignment:
            if ref_token is not None and hyp_token is not None and ref_token != hyp_token:
                # Check for phonetic similarity (simple approximation)
                if self.are_phonetically_similar(ref_token, hyp_token):
                    specific_errors["pronunciation_errors"].append((ref_token, hyp_token))
        
        # Analyze dictionary errors (words not in vocabulary)
        # This is a simplified approach; in practice, you'd check against a real dictionary
        for token in hyp_tokens:
            if token and len(token) > 2 and not self.is_common_word(token):
                specific_errors["dictionary_errors"].append(token)
        
        # Analyze suffix errors (wrong word form)
        for ref_token, hyp_token in alignment:
            if ref_token and hyp_token and ref_token != hyp_token:
                if self.is_suffix_error(ref_token, hyp_token):
                    specific_errors["suffix_errors"].append((ref_token, hyp_token))
        
        # Analyze nonsense errors (contextually inappropriate)
        specific_errors["nonsense_errors"] = self.detect_nonsense_errors(ref_tokens, hyp_tokens)
        
        # Analyze boundary errors (sentence boundary issues)
        specific_errors["boundary_errors"] = self.detect_boundary_errors(ref_tokens, hyp_tokens)
        
        # Analyze repetition errors
        specific_errors["repetition_errors"] = self.detect_repetitions(hyp_tokens)
        
        return specific_errors
    
    def are_phonetically_similar(self, word1: str, word2: str) -> bool:
        """
        Check if two words are phonetically similar.
        
        This is a simplified approach using edit distance. In a real system,
        you would use a phonetic algorithm like Soundex or Metaphone.
        
        Args:
            word1: First word.
            word2: Second word.
            
        Returns:
            True if words are phonetically similar, False otherwise.
        """
        # Simple heuristic: if edit distance is small relative to word length
        # and the first letter is the same, consider them phonetically similar
        if not word1 or not word2:
            return False
            
        max_length = max(len(word1), len(word2))
        if max_length == 0:
            return True
            
        distance = edit_distance(word1, word2)
        return (distance <= max_length / 3) and (word1[0] == word2[0])
    
    def is_common_word(self, word: str) -> bool:
        """
        Check if a word is common (simplified approach).
        
        Args:
            word: Word to check.
            
        Returns:
            True if the word is common, False otherwise.
        """
        # This is a placeholder. In a real system, you would check against a dictionary
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        return word.lower() in common_words
    
    def is_suffix_error(self, ref_word: str, hyp_word: str) -> bool:
        """
        Check if there's a suffix error between reference and hypothesis words.
        
        Args:
            ref_word: Reference word.
            hyp_word: Hypothesis word.
            
        Returns:
            True if there's a suffix error, False otherwise.
        """
        # Check if one word is a prefix of the other with a different suffix
        min_length = min(len(ref_word), len(hyp_word))
        if min_length < 3:
            return False
            
        # Check if they share a common stem (at least 3 characters)
        common_prefix_length = 0
        for i in range(min_length):
            if ref_word[i] == hyp_word[i]:
                common_prefix_length += 1
            else:
                break
                
        return common_prefix_length >= 3 and common_prefix_length < min(len(ref_word), len(hyp_word))
    
    def detect_nonsense_errors(self, ref_tokens: List[str], hyp_tokens: List[str]) -> List[str]:
        """
        Detect nonsense errors (contextually inappropriate words).
        
        Args:
            ref_tokens: Reference tokens.
            hyp_tokens: Hypothesis tokens.
            
        Returns:
            List of nonsense words.
        """
        # This is a simplified approach. In a real system, you would use
        # language models or contextual analysis.
        nonsense_words = []
        
        # Check for very short or very long words that are unusual
        for token in hyp_tokens:
            if token and (len(token) == 1 or len(token) > 15):
                if token not in ref_tokens and not token.isdigit():
                    nonsense_words.append(token)
        
        return nonsense_words
    
    def detect_boundary_errors(self, ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Detect sentence boundary errors.
        
        Args:
            ref_tokens: Reference tokens.
            hyp_tokens: Hypothesis tokens.
            
        Returns:
            List of boundary error descriptions.
        """
        # This is a simplified approach. In a real system, you would analyze
        # punctuation and sentence structure.
        boundary_errors = []
        
        # Check for missing sentence-final words
        if len(ref_tokens) >= 2 and len(hyp_tokens) >= 1:
            if ref_tokens[-1] != hyp_tokens[-1] and ref_tokens[-2] == hyp_tokens[-1]:
                boundary_errors.append(("missing_final", ref_tokens[-1]))
        
        # Check for missing sentence-initial words
        if len(ref_tokens) >= 2 and len(hyp_tokens) >= 1:
            if ref_tokens[0] != hyp_tokens[0] and ref_tokens[1] == hyp_tokens[0]:
                boundary_errors.append(("missing_initial", ref_tokens[0]))
        
        return boundary_errors
    
    def detect_repetitions(self, tokens: List[str]) -> List[str]:
        """
        Detect word repetitions.
        
        Args:
            tokens: List of tokens.
            
        Returns:
            List of repeated words.
        """
        repetitions = []
        
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                repetitions.append(tokens[i])
        
        return repetitions
    
    def analyze_all_transcripts(self) -> Dict[str, Any]:
        """
        Analyze errors in all transcript pairs.
        
        Returns:
            Dictionary containing comprehensive error analysis results.
        """
        logger.info("Analyzing errors in all transcript pairs")
        
        all_results = {}
        
        for file_path in self.common_keys:
            reference = self.reference_transcripts[file_path]
            hypothesis = self.hypothesis_transcripts[file_path]
            
            try:
                results = self.analyze_errors(reference, hypothesis)
                all_results[file_path] = results
                
                # Update global statistics
                self.error_stats["wer"][file_path] = results["wer"]
                self.error_stats["cer"][file_path] = results["cer"]
                
                # Update detailed error counts
                for ref_word, hyp_word in results["substitutions"]:
                    self.detailed_errors["substitutions"][(ref_word, hyp_word)] += 1
                
                for word in results["deletions"]:
                    self.detailed_errors["deletions"][word] += 1
                
                for word in results["insertions"]:
                    self.detailed_errors["insertions"][word] += 1
                
                # Add context for errors
                self.add_error_contexts(file_path, reference, hypothesis, results)
                
                logger.info(f"Analyzed {file_path}: WER={results['wer']:.4f}, CER={results['cer']:.4f}")
                
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {str(e)}")
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
        
        return all_results
    
    def add_error_contexts(self, file_path: str, reference: str, hypothesis: str, results: Dict[str, Any]) -> None:
        """
        Add context information for errors.
        
        Args:
            file_path: Path to the file being analyzed.
            reference: Reference transcription.
            hypothesis: Hypothesis transcription.
            results: Error analysis results.
        """
        # Add context for substitutions
        for ref_word, hyp_word in results["substitutions"]:
            context = self.get_error_context(reference, ref_word)
            self.detailed_errors["context_errors"].append({
                "file": file_path,
                "type": "substitution",
                "reference": ref_word,
                "hypothesis": hyp_word,
                "context": context
            })
        
        # Add context for deletions
        for word in results["deletions"]:
            context = self.get_error_context(reference, word)
            self.detailed_errors["context_errors"].append({
                "file": file_path,
                "type": "deletion",
                "reference": word,
                "hypothesis": None,
                "context": context
            })
        
        # Add context for insertions
        for word in results["insertions"]:
            context = self.get_error_context(hypothesis, word)
            self.detailed_errors["context_errors"].append({
                "file": file_path,
                "type": "insertion",
                "reference": None,
                "hypothesis": word,
                "context": context
            })
    
    def get_error_context(self, text: str, word: str, context_size: int = 2) -> str:
        """
        Get context around an error.
        
        Args:
            text: Full text.
            word: Word to find context for.
            context_size: Number of words to include on each side.
            
        Returns:
            Context string.
        """
        tokens = self.tokenize_text(text)
        
        for i, token in enumerate(tokens):
            if token == word:
                start = max(0, i - context_size)
                end = min(len(tokens), i + context_size + 1)
                context_tokens = tokens[start:end]
                return " ".join(context_tokens)
        
        return ""
    
    def calculate_summary_statistics(self) -> None:
        """
        Calculate summary statistics for all error types.
        """
        # Calculate average WER and CER
        self.summary_stats = {
            "avg_wer": np.mean(list(self.error_stats["wer"].values())),
            "avg_cer": np.mean(list(self.error_stats["cer"].values())),
            "total_files": len(self.common_keys)
        }
        
        # Count error types
        error_counts = {
            "substitutions": sum(len(results["substitutions"]) for results in self.error_stats.values() if isinstance(results, dict)),
            "deletions": sum(len(results["deletions"]) for results in self.error_stats.values() if isinstance(results, dict)),
            "insertions": sum(len(results["insertions"]) for results in self.error_stats.values() if isinstance(results, dict))
        }
        
        self.summary_stats.update(error_counts)
        
        # Most common errors
        self.summary_stats["common_substitutions"] = Counter(self.detailed_errors["substitutions"]).most_common(10)
        self.summary_stats["common_deletions"] = Counter(self.detailed_errors["deletions"]).most_common(10)
        self.summary_stats["common_insertions"] = Counter(self.detailed_errors["insertions"]).most_common(10)
    
    def generate_error_report(self, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive error analysis report.
        
        Args:
            output_dir: Directory to save the report. If None, the report is not saved.
            
        Returns:
            Dictionary containing the error report.
        """
        logger.info("Generating error analysis report")
        
        # Ensure we have analyzed the transcripts
        if not hasattr(self, 'summary_stats'):
            self.analyze_all_transcripts()
        
        # Create the report
        report = {
            "summary": self.summary_stats,
            "detailed_errors": {
                "substitutions": [{"reference": ref, "hypothesis": hyp, "count": count} 
                                 for (ref, hyp), count in Counter(self.detailed_errors["substitutions"]).most_common()],
                "deletions": [{"word": word, "count": count} 
                             for word, count in Counter(self.detailed_errors["deletions"]).most_common()],
                "insertions": [{"word": word, "count": count} 
                              for word, count in Counter(self.detailed_errors["insertions"]).most_common()],
                "contexts": self.detailed_errors["context_errors"]
            }
        }
        
        # Save the report if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "error_analysis_report.json")
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved error analysis report to {report_path}")
            
            # Generate visualizations
            self.generate_visualizations(output_dir)
        
        return report
    
    def generate_visualizations(self, output_dir: str) -> None:
        """
        Generate visualizations for error analysis.
        
        Args:
            output_dir: Directory to save visualizations.
        """
        logger.info("Generating error analysis visualizations")
        
        # Create error type distribution plot
        self.plot_error_distribution(output_dir)
        
        # Create WER distribution plot
        self.plot_wer_distribution(output_dir)
        
        # Create common errors plot
        self.plot_common_errors(output_dir)
    
    def plot_error_distribution(self, output_dir: str) -> None:
        """
        Plot distribution of error types.
        
        Args:
            output_dir: Directory to save the plot.
        """
        error_types = ['Substitutions', 'Deletions', 'Insertions']
        error_counts = [
            self.summary_stats["substitutions"],
            self.summary_stats["deletions"],
            self.summary_stats["insertions"]
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(error_types, error_counts, color=['blue', 'red', 'green'])
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Distribution of Error Types')
        plt.xlabel('Error Type')
        plt.ylabel('Count')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
        plt.close()
    
    def plot_wer_distribution(self, output_dir: str) -> None:
        """
        Plot distribution of WER across files.
        
        Args:
            output_dir: Directory to save the plot.
        """
        wer_values = list(self.error_stats["wer"].values())
        
        plt.figure(figsize=(10, 6))
        plt.hist(wer_values, bins=10, alpha=0.7, color='blue')
        plt.axvline(self.summary_stats["avg_wer"], color='red', linestyle='dashed', 
                   linewidth=2, label=f'Average WER: {self.summary_stats["avg_wer"]:.4f}')
        
        plt.title('Distribution of Word Error Rate (WER)')
        plt.xlabel('WER')
        plt.ylabel('Number of Files')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'wer_distribution.png'))
        plt.close()
    
    def plot_common_errors(self, output_dir: str) -> None:
        """
        Plot most common errors.
        
        Args:
            output_dir: Directory to save the plot.
        """
        # Plot common substitutions
        if self.summary_stats["common_substitutions"]:
            labels = [f"{ref}->{hyp}" for (ref, hyp), _ in self.summary_stats["common_substitutions"]]
            counts = [count for _, count in self.summary_stats["common_substitutions"]]
            
            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels, counts, color='blue', alpha=0.7)
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center')
            
            plt.title('Most Common Substitution Errors')
            plt.xlabel('Count')
            plt.ylabel('Substitution')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'common_substitutions.png'))
            plt.close()
        
        # Plot common deletions
        if self.summary_stats["common_deletions"]:
            labels = [word for word, _ in self.summary_stats["common_deletions"]]
            counts = [count for _, count in self.summary_stats["common_deletions"]]
            
            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels, counts, color='red', alpha=0.7)
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center')
            
            plt.title('Most Common Deletion Errors')
            plt.xlabel('Count')
            plt.ylabel('Deleted Word')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'common_deletions.png'))
            plt.close()
        
        # Plot common insertions
        if self.summary_stats["common_insertions"]:
            labels = [word for word, _ in self.summary_stats["common_insertions"]]
            counts = [count for _, count in self.summary_stats["common_insertions"]]
            
            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels, counts, color='green', alpha=0.7)
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                        f'{int(width)}', ha='left', va='center')
            
            plt.title('Most Common Insertion Errors')
            plt.xlabel('Count')
            plt.ylabel('Inserted Word')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'common_insertions.png'))
            plt.close()
