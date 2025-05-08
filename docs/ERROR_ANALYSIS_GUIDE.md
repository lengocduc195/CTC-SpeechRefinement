# Speech Recognition Error Analysis Guide

This guide explains how to use the error analysis modules to analyze and improve speech recognition results.

## Overview

The error analysis system consists of three main modules:

1. **ErrorAnalyzer**: Identifies and categorizes errors by comparing reference and hypothesis transcriptions.
2. **ErrorVisualizer**: Creates visualizations to help understand error patterns.
3. **ErrorImprover**: Analyzes error patterns and suggests improvements.

## Error Types

The system analyzes the following types of speech recognition errors:

### 1. Substitution Errors

Substitution errors occur when a word in the reference transcription is replaced with a different word in the recognition result.

**Example**: "1-2-3" recognized as "1-5-3"

**Common causes**:
- Phonetically similar words (homonyms)
- Suboptimal language model
- Acoustic similarity
- Short word confusion

### 2. Deletion Errors

Deletion errors occur when a word in the reference transcription is omitted in the recognition result.

**Example**: "1-2-3" recognized as "1-3"

**Common causes**:
- Speech misdetection
- Low confidence rejection
- Word merging (e.g., "four two" → "forty")
- Short or quiet words

### 3. Insertion Errors

Insertion errors occur when a word appears in the recognition result but is not in the reference transcription.

**Common causes**:
- Noise misinterpreted as speech
- Repetition of words
- Filler words
- Segmentation issues

### 4. Specific Error Patterns

The system also identifies more specific error patterns:

- **Pronunciation errors**: Confusion between similar-sounding words
- **Dictionary errors**: Words not in the recognition vocabulary
- **Suffix errors**: Incorrect word forms (e.g., tense, plurality)
- **Nonsense errors**: Contextually inappropriate words
- **Boundary errors**: Sentence boundary issues
- **Repetition errors**: Unintended word repetitions

## Using the Error Analysis System

### Running Error Analysis on test2 Data

To analyze errors in the test2 dataset, run:

```bash
python run_test2_error_analysis.py --reference_dir data/test2/reference --hypothesis_dir transcripts/test2 --output_dir results/test2/error_analysis
```

#### Command-line Arguments

- `--reference_dir`: Directory containing reference transcriptions
- `--hypothesis_dir`: Directory containing hypothesis (recognized) transcriptions
- `--output_dir`: Directory to save error analysis results
- `--skip_visualizations`: Optional flag to skip generating visualizations

### Output Files

The error analysis generates the following output files:

1. **error_analysis_report.json**: Detailed error analysis results
2. **error_patterns.json**: Identified error patterns
3. **improvement_suggestions.md**: Suggested improvements to address errors
4. **visualizations/**: Directory containing error visualizations

## Understanding the Results

### Error Analysis Report

The error analysis report contains:

- **Summary statistics**: WER, CER, total errors by type
- **Detailed errors**: Lists of substitutions, deletions, and insertions
- **File-level statistics**: Error rates for each file

### Error Patterns

The error patterns analysis identifies:

- **Substitution patterns**: Words commonly substituted with specific alternatives
- **Deletion patterns**: Words commonly deleted in specific contexts
- **Insertion patterns**: Words commonly inserted in specific contexts
- **Phonetic confusions**: Words confused due to phonetic similarity
- **Context patterns**: Error patterns related to specific contexts

### Improvement Suggestions

The improvement suggestions include:

- **Acoustic model improvements**: Ways to improve the acoustic model
- **Language model improvements**: Ways to improve the language model
- **Preprocessing improvements**: Ways to improve audio preprocessing
- **Postprocessing improvements**: Ways to improve result postprocessing
- **Specific word fixes**: Fixes for specific problematic words

### Visualizations

The visualizations include:

- **Error type distribution**: Pie chart showing the distribution of error types
- **Word clouds**: Word clouds for different error types
- **Substitution network**: Network graph of substitution errors
- **Error heatmap**: Heatmap of error rates by file
- **Error by word length**: Bar chart of errors by word length

## Using the Modules Programmatically

### ErrorAnalyzer

```python
from ctc_speech_refinement.core.eda.error_analysis import ErrorAnalyzer

# Initialize analyzer
analyzer = ErrorAnalyzer(
    reference_transcripts=reference_transcripts,
    hypothesis_transcripts=hypothesis_transcripts
)

# Analyze errors
analyzer.analyze_all_transcripts()

# Generate report
report = analyzer.generate_error_report(output_dir)
```

### ErrorVisualizer

```python
from ctc_speech_refinement.core.eda.error_visualization import ErrorVisualizer

# Initialize visualizer
visualizer = ErrorVisualizer(report)

# Create visualizations
visualizer.create_visualizations(output_dir)
```

### ErrorImprover

```python
from ctc_speech_refinement.core.eda.error_improvement import ErrorImprover

# Initialize improver
improver = ErrorImprover(report)

# Analyze error patterns
error_patterns = improver.analyze_error_patterns()

# Generate and save improvement suggestions
suggestions_path = improver.save_suggestions(output_dir)
```

## Implementing Improvements

Based on the error analysis, you can implement improvements such as:

### Acoustic Model Improvements

- Fine-tune the acoustic model with examples of commonly confused words
- Adjust model sensitivity for short words
- Train with more diverse acoustic conditions

### Language Model Improvements

- Enhance the language model with domain-specific data
- Increase the weight of the language model for commonly deleted words
- Add context-specific language model rules

### Preprocessing Improvements

- Improve noise filtering
- Adjust silence detection thresholds
- Enhance low-volume speech segments

### Postprocessing Improvements

- Implement rules to correct common confusions
- Add post-processing to detect and remove unintended word repetitions
- Implement confidence scoring to flag potential errors

## Example Workflow

1. **Run transcription** on test2 data
2. **Analyze errors** using the error analysis system
3. **Review error patterns** and improvement suggestions
4. **Implement improvements** based on the analysis
5. **Re-run transcription** to verify improvements
6. **Compare results** to measure the impact of improvements

## Conclusion

The error analysis system provides a comprehensive framework for understanding and addressing speech recognition errors. By systematically analyzing error patterns and implementing targeted improvements, you can significantly enhance the accuracy of your speech recognition system.
