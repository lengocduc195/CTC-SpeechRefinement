"""
Error improvement module for speech recognition results.

This module provides functions to analyze error patterns and suggest improvements
for speech recognition systems.
"""

import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import re
import numpy as np
import pandas as pd
from Levenshtein import distance

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorImprover:
    """
    Analyzer for speech recognition error patterns and improvement suggestions.

    This class provides methods to analyze error patterns and suggest improvements
    for speech recognition systems.
    """

    def __init__(self, error_report: Dict[str, Any]):
        """
        Initialize the ErrorImprover.

        Args:
            error_report: Error analysis report from ErrorAnalyzer.
        """
        self.error_report = error_report

        # Initialize error pattern analysis
        self.error_patterns = {
            "substitution_patterns": [],
            "deletion_patterns": [],
            "insertion_patterns": [],
            "phonetic_confusions": [],
            "context_patterns": []
        }

        logger.info("Initialized ErrorImprover")

    def analyze_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze error patterns in the error report.

        Returns:
            Dictionary containing error pattern analysis.
        """
        logger.info("Analyzing error patterns")

        # Analyze substitution patterns
        self._analyze_substitution_patterns()

        # Analyze deletion patterns
        self._analyze_deletion_patterns()

        # Analyze insertion patterns
        self._analyze_insertion_patterns()

        # Analyze phonetic confusions
        self._analyze_phonetic_confusions()

        # Analyze context patterns
        self._analyze_context_patterns()

        return self.error_patterns

    def _analyze_substitution_patterns(self) -> None:
        """
        Analyze patterns in substitution errors.
        """
        logger.info("Analyzing substitution patterns")

        substitutions = self.error_report['detailed_errors'].get('substitutions', [])

        if not substitutions:
            logger.warning("No substitution data to analyze")
            return

        # Group substitutions by reference word
        ref_word_subs = defaultdict(list)
        for sub in substitutions:
            ref = sub['reference']
            hyp = sub['hypothesis']
            count = sub['count']
            ref_word_subs[ref].append((hyp, count))

        # Identify words with multiple substitutions
        multiple_subs = {}
        for ref, subs in ref_word_subs.items():
            if len(subs) > 1:
                multiple_subs[ref] = subs

        # Analyze patterns
        for ref, subs in multiple_subs.items():
            total_count = sum(count for _, count in subs)
            pattern = {
                "reference": ref,
                "substitutions": [{"hypothesis": hyp, "count": count, "percentage": count/total_count*100}
                                 for hyp, count in subs],
                "total_count": total_count,
                "pattern_type": self._identify_substitution_pattern_type(ref, [hyp for hyp, _ in subs])
            }
            self.error_patterns["substitution_patterns"].append(pattern)

        # Sort by total count
        self.error_patterns["substitution_patterns"].sort(key=lambda x: x["total_count"], reverse=True)

    def _identify_substitution_pattern_type(self, reference: str, hypotheses: List[str]) -> str:
        """
        Identify the type of substitution pattern.

        Args:
            reference: Reference word.
            hypotheses: List of hypothesis words.

        Returns:
            Pattern type description.
        """
        # Check for phonetic similarity
        phonetic_similar = all(self._are_phonetically_similar(reference, hyp) for hyp in hypotheses)
        if phonetic_similar:
            return "Phonetic confusion"

        # Check for similar length
        length_similar = all(abs(len(reference) - len(hyp)) <= 1 for hyp in hypotheses)
        if length_similar:
            return "Similar length confusion"

        # Check for short word
        if len(reference) <= 2:
            return "Short word confusion"

        # Default
        return "Mixed confusion patterns"

    def _are_phonetically_similar(self, word1: str, word2: str) -> bool:
        """
        Check if two words are phonetically similar.

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

        edit_dist = distance(word1, word2)
        return (edit_dist <= max_length / 3) and (word1[0] == word2[0])

    def _analyze_deletion_patterns(self) -> None:
        """
        Analyze patterns in deletion errors.
        """
        logger.info("Analyzing deletion patterns")

        deletions = self.error_report['detailed_errors'].get('deletions', [])

        if not deletions:
            logger.warning("No deletion data to analyze")
            return

        # Group deletions by word length
        deletions_by_length = defaultdict(list)
        for deletion in deletions:
            word = deletion['word']
            count = deletion['count']
            deletions_by_length[len(word)].append((word, count))

        # Analyze patterns by word length
        for length, words in deletions_by_length.items():
            total_count = sum(count for _, count in words)
            pattern = {
                "word_length": length,
                "examples": [{"word": word, "count": count} for word, count in sorted(words, key=lambda x: x[1], reverse=True)[:5]],
                "total_count": total_count,
                "pattern_type": self._identify_deletion_pattern_type(length, [word for word, _ in words])
            }
            self.error_patterns["deletion_patterns"].append(pattern)

        # Sort by total count
        self.error_patterns["deletion_patterns"].sort(key=lambda x: x["total_count"], reverse=True)

    def _identify_deletion_pattern_type(self, word_length: int, words: List[str]) -> str:
        """
        Identify the type of deletion pattern.

        Args:
            word_length: Length of deleted words.
            words: List of deleted words.

        Returns:
            Pattern type description.
        """
        # Check for short words
        if word_length <= 2:
            return "Short word deletion"

        # Check for function words
        function_words = {"the", "a", "an", "of", "to", "in", "for", "on", "with", "by", "at"}
        function_word_ratio = sum(1 for word in words if word.lower() in function_words) / len(words)
        if function_word_ratio > 0.5:
            return "Function word deletion"

        # Default
        return "Content word deletion"

    def _analyze_insertion_patterns(self) -> None:
        """
        Analyze patterns in insertion errors.
        """
        logger.info("Analyzing insertion patterns")

        insertions = self.error_report['detailed_errors'].get('insertions', [])

        if not insertions:
            logger.warning("No insertion data to analyze")
            return

        # Group insertions by word length
        insertions_by_length = defaultdict(list)
        for insertion in insertions:
            word = insertion['word']
            count = insertion['count']
            insertions_by_length[len(word)].append((word, count))

        # Analyze patterns by word length
        for length, words in insertions_by_length.items():
            total_count = sum(count for _, count in words)
            pattern = {
                "word_length": length,
                "examples": [{"word": word, "count": count} for word, count in sorted(words, key=lambda x: x[1], reverse=True)[:5]],
                "total_count": total_count,
                "pattern_type": self._identify_insertion_pattern_type(length, [word for word, _ in words])
            }
            self.error_patterns["insertion_patterns"].append(pattern)

        # Sort by total count
        self.error_patterns["insertion_patterns"].sort(key=lambda x: x["total_count"], reverse=True)

    def _identify_insertion_pattern_type(self, word_length: int, words: List[str]) -> str:
        """
        Identify the type of insertion pattern.

        Args:
            word_length: Length of inserted words.
            words: List of inserted words.

        Returns:
            Pattern type description.
        """
        # Check for short words
        if word_length <= 2:
            return "Short word insertion"

        # Check for filler words
        filler_words = {"um", "uh", "like", "so", "well", "actually"}
        filler_word_ratio = sum(1 for word in words if word.lower() in filler_words) / len(words)
        if filler_word_ratio > 0.3:
            return "Filler word insertion"

        # Default
        return "Content word insertion"

    def _analyze_phonetic_confusions(self) -> None:
        """
        Analyze phonetic confusions in substitution errors.
        """
        logger.info("Analyzing phonetic confusions")

        substitutions = self.error_report['detailed_errors'].get('substitutions', [])

        if not substitutions:
            logger.warning("No substitution data to analyze phonetic confusions")
            return

        # Find phonetically similar substitutions
        phonetic_confusions = []
        for sub in substitutions:
            ref = sub['reference']
            hyp = sub['hypothesis']
            count = sub['count']

            if self._are_phonetically_similar(ref, hyp):
                phonetic_confusions.append({
                    "reference": ref,
                    "hypothesis": hyp,
                    "count": count,
                    "edit_distance": distance(ref, hyp),
                    "confusion_type": self._identify_phonetic_confusion_type(ref, hyp)
                })

        # Sort by count
        phonetic_confusions.sort(key=lambda x: x["count"], reverse=True)

        self.error_patterns["phonetic_confusions"] = phonetic_confusions

    def _identify_phonetic_confusion_type(self, reference: str, hypothesis: str) -> str:
        """
        Identify the type of phonetic confusion.

        Args:
            reference: Reference word.
            hypothesis: Hypothesis word.

        Returns:
            Confusion type description.
        """
        # Check for vowel confusion
        vowels = "aeiou"
        ref_vowels = [c for c in reference if c.lower() in vowels]
        hyp_vowels = [c for c in hypothesis if c.lower() in vowels]

        if len(ref_vowels) != len(hyp_vowels):
            return "Vowel count difference"

        # Check for consonant confusion
        ref_consonants = [c for c in reference if c.lower() not in vowels and c.isalpha()]
        hyp_consonants = [c for c in hypothesis if c.lower() not in vowels and c.isalpha()]

        if len(ref_consonants) != len(hyp_consonants):
            return "Consonant count difference"

        # Check for similar sounding consonants
        similar_consonants = {
            'b': ['p', 'v'],
            'd': ['t', 'th'],
            'g': ['k', 'c'],
            'v': ['f', 'b'],
            'z': ['s'],
            'm': ['n'],
            'n': ['m'],
            'l': ['r'],
            'r': ['l'],
            'th': ['d', 't'],
            'sh': ['ch'],
            'ch': ['sh', 'j'],
            'j': ['ch']
        }

        for i, c in enumerate(reference):
            if i < len(hypothesis) and c != hypothesis[i]:
                if c.lower() in similar_consonants and hypothesis[i].lower() in similar_consonants.get(c.lower(), []):
                    return f"Similar consonant confusion ({c}/{hypothesis[i]})"

        # Default
        return "General phonetic similarity"

    def _analyze_context_patterns(self) -> None:
        """
        Analyze context patterns in errors.
        """
        logger.info("Analyzing context patterns")

        contexts = self.error_report['detailed_errors'].get('contexts', [])

        if not contexts:
            logger.warning("No context data to analyze")
            return

        # Analyze context patterns
        # This is a simplified approach; in a real system, you would use more sophisticated
        # natural language processing techniques

        # Group errors by context
        context_errors = defaultdict(list)
        for context_error in contexts:
            context = context_error.get('context', '')
            error_type = context_error.get('type', '')
            ref = context_error.get('reference', '')
            hyp = context_error.get('hypothesis', '')

            # Create a simplified context by keeping only a few words around the error
            simplified_context = self._simplify_context(context, ref if ref else hyp)

            context_errors[simplified_context].append({
                "type": error_type,
                "reference": ref,
                "hypothesis": hyp
            })

        # Find repeated context patterns
        repeated_contexts = []
        for context, errors in context_errors.items():
            if len(errors) > 1:
                repeated_contexts.append({
                    "context": context,
                    "errors": errors,
                    "count": len(errors)
                })

        # Sort by count
        repeated_contexts.sort(key=lambda x: x["count"], reverse=True)

        self.error_patterns["context_patterns"] = repeated_contexts[:10]  # Limit to top 10

    def _simplify_context(self, context: str, error_word: str) -> str:
        """
        Simplify a context string to focus on words around the error.

        Args:
            context: Full context string.
            error_word: The error word to find in the context.

        Returns:
            Simplified context string.
        """
        if not error_word or not context:
            return context

        words = context.split()
        if error_word not in words:
            return context

        error_idx = words.index(error_word)
        start_idx = max(0, error_idx - 1)
        end_idx = min(len(words), error_idx + 2)

        return " ".join(words[start_idx:end_idx])

    def generate_improvement_suggestions(self) -> Dict[str, Any]:
        """
        Generate improvement suggestions based on error patterns.

        Returns:
            Dictionary containing improvement suggestions.
        """
        logger.info("Generating improvement suggestions")

        # Ensure we have analyzed error patterns
        if not self.error_patterns["substitution_patterns"] and not self.error_patterns["phonetic_confusions"]:
            self.analyze_error_patterns()

        # Generate suggestions
        suggestions = {
            "acoustic_model_improvements": self._suggest_acoustic_model_improvements(),
            "language_model_improvements": self._suggest_language_model_improvements(),
            "preprocessing_improvements": self._suggest_preprocessing_improvements(),
            "postprocessing_improvements": self._suggest_postprocessing_improvements(),
            "specific_word_fixes": self._suggest_specific_word_fixes()
        }

        return suggestions

    def _suggest_acoustic_model_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements for the acoustic model.

        Returns:
            List of improvement suggestions.
        """
        suggestions = []

        # Analyze phonetic confusions
        if self.error_patterns["phonetic_confusions"]:
            phonetic_groups = defaultdict(list)
            for confusion in self.error_patterns["phonetic_confusions"]:
                confusion_type = confusion["confusion_type"]
                phonetic_groups[confusion_type].append(confusion)

            # Generate suggestions for each confusion type
            for confusion_type, confusions in phonetic_groups.items():
                examples = [f"'{c['reference']}' → '{c['hypothesis']}'" for c in confusions[:3]]
                suggestion = {
                    "issue": f"Phonetic confusion: {confusion_type}",
                    "examples": examples,
                    "suggestion": f"Fine-tune acoustic model with examples of {confusion_type} confusions",
                    "priority": "high" if len(confusions) > 5 else "medium"
                }
                suggestions.append(suggestion)

        # Analyze deletion patterns
        if self.error_patterns["deletion_patterns"]:
            for pattern in self.error_patterns["deletion_patterns"][:3]:
                pattern_type = pattern["pattern_type"]
                examples = [f"'{e['word']}'" for e in pattern["examples"][:3]]
                suggestion = {
                    "issue": f"Deletion pattern: {pattern_type}",
                    "examples": examples,
                    "suggestion": self._get_deletion_suggestion(pattern_type),
                    "priority": "high" if pattern["total_count"] > 10 else "medium"
                }
                suggestions.append(suggestion)

        return suggestions

    def _get_deletion_suggestion(self, pattern_type: str) -> str:
        """
        Get suggestion for a deletion pattern type.

        Args:
            pattern_type: Type of deletion pattern.

        Returns:
            Suggestion string.
        """
        if pattern_type == "Short word deletion":
            return "Adjust acoustic model to better detect short words; consider lowering detection threshold"
        elif pattern_type == "Function word deletion":
            return "Train acoustic model with more examples of function words in various contexts"
        else:
            return "Review acoustic model performance on content words; may need more diverse training data"

    def _suggest_language_model_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements for the language model.

        Returns:
            List of improvement suggestions.
        """
        suggestions = []

        # Analyze substitution patterns
        if self.error_patterns["substitution_patterns"]:
            for pattern in self.error_patterns["substitution_patterns"][:3]:
                examples = [f"'{pattern['reference']}' → '{s['hypothesis']}'" for s in pattern["substitutions"][:3]]
                suggestion = {
                    "issue": f"Substitution pattern for '{pattern['reference']}'",
                    "examples": examples,
                    "suggestion": "Enhance language model with more context examples containing this word",
                    "priority": "high" if pattern["total_count"] > 10 else "medium"
                }
                suggestions.append(suggestion)

        # Analyze context patterns
        if self.error_patterns["context_patterns"]:
            for pattern in self.error_patterns["context_patterns"][:3]:
                context = pattern["context"]
                error_types = Counter([e["type"] for e in pattern["errors"]])
                most_common_type = error_types.most_common(1)[0][0]

                suggestion = {
                    "issue": f"Context pattern with {most_common_type} errors",
                    "examples": [context],
                    "suggestion": "Improve language model's handling of this specific context pattern",
                    "priority": "medium"
                }
                suggestions.append(suggestion)

        return suggestions

    def _suggest_preprocessing_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements for audio preprocessing.

        Returns:
            List of improvement suggestions.
        """
        suggestions = []

        # Check for insertion patterns that might be noise-related
        if self.error_patterns["insertion_patterns"]:
            noise_related = [p for p in self.error_patterns["insertion_patterns"]
                            if p["word_length"] <= 2 or p["pattern_type"] == "Short word insertion"]

            if noise_related:
                suggestion = {
                    "issue": "Possible noise-related insertions",
                    "examples": [f"'{e['word']}'" for p in noise_related[:2] for e in p["examples"][:2]],
                    "suggestion": "Improve noise filtering in preprocessing; consider adjusting voice activity detection",
                    "priority": "high" if sum(p["total_count"] for p in noise_related) > 10 else "medium"
                }
                suggestions.append(suggestion)

        # Check for deletion patterns that might be related to audio quality
        if self.error_patterns["deletion_patterns"]:
            quality_related = [p for p in self.error_patterns["deletion_patterns"]
                              if p["pattern_type"] == "Short word deletion"]

            if quality_related:
                suggestion = {
                    "issue": "Possible audio quality issues causing deletions",
                    "examples": [f"'{e['word']}'" for p in quality_related[:2] for e in p["examples"][:2]],
                    "suggestion": "Review audio preprocessing pipeline; consider enhancing low-volume segments",
                    "priority": "medium"
                }
                suggestions.append(suggestion)

        return suggestions

    def _suggest_postprocessing_improvements(self) -> List[Dict[str, Any]]:
        """
        Suggest improvements for postprocessing.

        Returns:
            List of improvement suggestions.
        """
        suggestions = []

        # Suggest specific substitution corrections
        if self.error_patterns["phonetic_confusions"]:
            top_confusions = self.error_patterns["phonetic_confusions"][:5]

            suggestion = {
                "issue": "Common phonetic confusions",
                "examples": [f"'{c['reference']}' → '{c['hypothesis']}'" for c in top_confusions],
                "suggestion": "Implement post-processing rules to correct these common confusions",
                "priority": "high"
            }
            suggestions.append(suggestion)

        # Suggest handling for insertions
        if self.error_patterns["insertion_patterns"]:
            repeated_words = [p for p in self.error_patterns["insertion_patterns"]
                             if any(e["word"] == e["word"] for e in p["examples"])]

            if repeated_words:
                suggestion = {
                    "issue": "Word repetition insertions",
                    "examples": [f"'{e['word']}'" for p in repeated_words[:2] for e in p["examples"][:2]],
                    "suggestion": "Implement post-processing to detect and remove unintended word repetitions",
                    "priority": "medium"
                }
                suggestions.append(suggestion)

        return suggestions

    def _suggest_specific_word_fixes(self) -> List[Dict[str, Any]]:
        """
        Suggest fixes for specific problematic words.

        Returns:
            List of specific word fix suggestions.
        """
        suggestions = []

        # Combine all error types to find the most problematic words
        word_issues = defaultdict(int)

        # Count substitution issues
        for sub in self.error_report['detailed_errors'].get('substitutions', []):
            word_issues[sub['reference']] += sub['count']

        # Count deletion issues
        for deletion in self.error_report['detailed_errors'].get('deletions', []):
            word_issues[deletion['word']] += deletion['count']

        # Find top problematic words
        top_words = sorted(word_issues.items(), key=lambda x: x[1], reverse=True)[:10]

        # Generate suggestions for top words
        for word, count in top_words:
            # Find all errors related to this word
            word_subs = [sub for sub in self.error_report['detailed_errors'].get('substitutions', [])
                        if sub['reference'] == word]

            word_deletions = [deletion for deletion in self.error_report['detailed_errors'].get('deletions', [])
                             if deletion['word'] == word]

            # Determine the main issue type
            if word_subs and (not word_deletions or word_subs[0]['count'] > word_deletions[0]['count']):
                issue_type = "substitution"
                examples = [f"'{word}' → '{sub['hypothesis']}'" for sub in word_subs[:3]]
            else:
                issue_type = "deletion"
                examples = [f"'{word}' deleted" for _ in range(min(3, len(word_deletions)))]

            suggestion = {
                "word": word,
                "issue_type": issue_type,
                "count": count,
                "examples": examples,
                "suggestion": self._get_word_specific_suggestion(word, issue_type),
                "priority": "high" if count > 10 else "medium"
            }
            suggestions.append(suggestion)

        return suggestions

    def _get_word_specific_suggestion(self, word: str, issue_type: str) -> str:
        """
        Get a suggestion for a specific word issue.

        Args:
            word: The problematic word.
            issue_type: Type of issue (substitution or deletion).

        Returns:
            Suggestion string.
        """
        if issue_type == "substitution":
            if len(word) <= 2:
                return f"Add '{word}' to a custom dictionary with increased weight; consider context-specific language model rules"
            else:
                return f"Add more training examples with '{word}' in various contexts; consider pronunciation variants"
        else:  # deletion
            if len(word) <= 2:
                return f"Lower detection threshold for short words like '{word}'; add post-processing rules to insert in common contexts"
            else:
                return f"Review audio examples where '{word}' is deleted; may need acoustic model fine-tuning"

    def save_suggestions(self, output_dir: str) -> str:
        """
        Save improvement suggestions to a file.

        Args:
            output_dir: Directory to save the suggestions.

        Returns:
            Path to the saved file.
        """
        logger.info("Saving improvement suggestions")

        # Generate suggestions
        suggestions = self.generate_improvement_suggestions()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save as JSON
        json_path = os.path.join(output_dir, "improvement_suggestions.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(suggestions, f, indent=2, ensure_ascii=False)

        # Save as Markdown
        md_path = os.path.join(output_dir, "improvement_suggestions.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Speech Recognition Improvement Suggestions\n\n")

            # Add summary
            f.write("## Summary\n\n")
            f.write(f"Based on analysis of {self.error_report['summary']['total_files']} files with:\n")
            f.write(f"- Average WER: {self.error_report['summary']['avg_wer']:.4f}\n")
            f.write(f"- Total substitutions: {self.error_report['summary']['substitutions']}\n")
            f.write(f"- Total deletions: {self.error_report['summary']['deletions']}\n")
            f.write(f"- Total insertions: {self.error_report['summary']['insertions']}\n\n")

            # Add acoustic model suggestions
            f.write("## Acoustic Model Improvements\n\n")
            for suggestion in suggestions["acoustic_model_improvements"]:
                f.write(f"### {suggestion['issue']} (Priority: {suggestion['priority']})\n\n")
                f.write("**Examples:**\n")
                for example in suggestion["examples"]:
                    f.write(f"- {example}\n")
                f.write(f"\n**Suggestion:** {suggestion['suggestion']}\n\n")

            # Add language model suggestions
            f.write("## Language Model Improvements\n\n")
            for suggestion in suggestions["language_model_improvements"]:
                f.write(f"### {suggestion['issue']} (Priority: {suggestion['priority']})\n\n")
                f.write("**Examples:**\n")
                for example in suggestion["examples"]:
                    f.write(f"- {example}\n")
                f.write(f"\n**Suggestion:** {suggestion['suggestion']}\n\n")

            # Add preprocessing suggestions
            f.write("## Preprocessing Improvements\n\n")
            for suggestion in suggestions["preprocessing_improvements"]:
                f.write(f"### {suggestion['issue']} (Priority: {suggestion['priority']})\n\n")
                f.write("**Examples:**\n")
                for example in suggestion["examples"]:
                    f.write(f"- {example}\n")
                f.write(f"\n**Suggestion:** {suggestion['suggestion']}\n\n")

            # Add postprocessing suggestions
            f.write("## Postprocessing Improvements\n\n")
            for suggestion in suggestions["postprocessing_improvements"]:
                f.write(f"### {suggestion['issue']} (Priority: {suggestion['priority']})\n\n")
                f.write("**Examples:**\n")
                for example in suggestion["examples"]:
                    f.write(f"- {example}\n")
                f.write(f"\n**Suggestion:** {suggestion['suggestion']}\n\n")

            # Add specific word fixes
            f.write("## Specific Word Fixes\n\n")
            f.write("| Word | Issue Type | Count | Suggestion | Priority |\n")
            f.write("|------|------------|-------|------------|----------|\n")
            for suggestion in suggestions["specific_word_fixes"]:
                f.write(f"| '{suggestion['word']}' | {suggestion['issue_type']} | {suggestion['count']} | {suggestion['suggestion']} | {suggestion['priority']} |\n")

        logger.info(f"Saved improvement suggestions to {md_path}")
        return md_path
