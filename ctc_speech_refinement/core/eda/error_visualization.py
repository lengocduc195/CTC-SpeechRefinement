"""
Error visualization module for speech recognition results.

This module provides functions to visualize error patterns in speech recognition results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
import networkx as nx
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ErrorVisualizer:
    """
    Visualizer for speech recognition errors.
    
    This class provides methods to visualize error patterns in speech recognition results.
    """
    
    def __init__(self, error_report: Dict[str, Any]):
        """
        Initialize the ErrorVisualizer.
        
        Args:
            error_report: Error analysis report from ErrorAnalyzer.
        """
        self.error_report = error_report
        logger.info("Initialized ErrorVisualizer")
    
    def create_visualizations(self, output_dir: str) -> None:
        """
        Create all visualizations.
        
        Args:
            output_dir: Directory to save visualizations.
        """
        logger.info("Creating error visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create error type distribution
        self.plot_error_type_distribution(output_dir)
        
        # Create error word clouds
        self.create_error_word_clouds(output_dir)
        
        # Create substitution network graph
        self.create_substitution_network(output_dir)
        
        # Create error heatmap
        self.create_error_heatmap(output_dir)
        
        # Create error patterns by word length
        self.plot_error_by_word_length(output_dir)
        
        logger.info(f"All visualizations saved to {output_dir}")
    
    def plot_error_type_distribution(self, output_dir: str) -> None:
        """
        Plot distribution of error types with percentages.
        
        Args:
            output_dir: Directory to save the plot.
        """
        logger.info("Plotting error type distribution")
        
        # Get error counts
        substitutions = self.error_report['summary']['substitutions']
        deletions = self.error_report['summary']['deletions']
        insertions = self.error_report['summary']['insertions']
        
        total_errors = substitutions + deletions + insertions
        
        # Calculate percentages
        sub_pct = substitutions / total_errors * 100 if total_errors > 0 else 0
        del_pct = deletions / total_errors * 100 if total_errors > 0 else 0
        ins_pct = insertions / total_errors * 100 if total_errors > 0 else 0
        
        # Create pie chart
        plt.figure(figsize=(10, 7))
        labels = [f'Substitutions\n{substitutions} ({sub_pct:.1f}%)', 
                 f'Deletions\n{deletions} ({del_pct:.1f}%)', 
                 f'Insertions\n{insertions} ({ins_pct:.1f}%)']
        sizes = [substitutions, deletions, insertions]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0, 0)  # explode the 1st slice (Substitutions)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Distribution of Error Types', fontsize=16)
        
        plt.savefig(os.path.join(output_dir, 'error_type_pie_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_word_clouds(self, output_dir: str) -> None:
        """
        Create word clouds for different error types.
        
        Args:
            output_dir: Directory to save the word clouds.
        """
        logger.info("Creating error word clouds")
        
        # Create word cloud for substitution reference words
        sub_ref_words = {}
        for item in self.error_report['detailed_errors']['substitutions']:
            sub_ref_words[item['reference']] = sub_ref_words.get(item['reference'], 0) + item['count']
        
        if sub_ref_words:
            self._create_word_cloud(sub_ref_words, 
                                   os.path.join(output_dir, 'substitution_reference_wordcloud.png'),
                                   'Commonly Misrecognized Words (Substitution Errors)',
                                   colormap='Reds')
        
        # Create word cloud for substitution hypothesis words
        sub_hyp_words = {}
        for item in self.error_report['detailed_errors']['substitutions']:
            sub_hyp_words[item['hypothesis']] = sub_hyp_words.get(item['hypothesis'], 0) + item['count']
        
        if sub_hyp_words:
            self._create_word_cloud(sub_hyp_words, 
                                   os.path.join(output_dir, 'substitution_hypothesis_wordcloud.png'),
                                   'Words Incorrectly Substituted',
                                   colormap='Blues')
        
        # Create word cloud for deletion words
        del_words = {}
        for item in self.error_report['detailed_errors']['deletions']:
            del_words[item['word']] = item['count']
        
        if del_words:
            self._create_word_cloud(del_words, 
                                   os.path.join(output_dir, 'deletion_wordcloud.png'),
                                   'Commonly Deleted Words',
                                   colormap='Oranges')
        
        # Create word cloud for insertion words
        ins_words = {}
        for item in self.error_report['detailed_errors']['insertions']:
            ins_words[item['word']] = item['count']
        
        if ins_words:
            self._create_word_cloud(ins_words, 
                                   os.path.join(output_dir, 'insertion_wordcloud.png'),
                                   'Commonly Inserted Words',
                                   colormap='Greens')
    
    def _create_word_cloud(self, word_freq: Dict[str, int], output_path: str, 
                          title: str, colormap: str = 'viridis') -> None:
        """
        Create a word cloud from word frequencies.
        
        Args:
            word_freq: Dictionary mapping words to frequencies.
            output_path: Path to save the word cloud.
            title: Title of the word cloud.
            colormap: Matplotlib colormap name.
        """
        if not word_freq:
            logger.warning(f"No words to create word cloud for {title}")
            return
            
        plt.figure(figsize=(12, 8))
        wc = WordCloud(width=1200, height=800, background_color='white', 
                      colormap=colormap, max_words=100, contour_width=1, contour_color='black')
        
        # Generate word cloud
        wc.generate_from_frequencies(word_freq)
        
        # Display the word cloud
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16)
        plt.tight_layout(pad=0)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_substitution_network(self, output_dir: str, max_edges: int = 30) -> None:
        """
        Create a network graph of substitution errors.
        
        Args:
            output_dir: Directory to save the graph.
            max_edges: Maximum number of edges to include.
        """
        logger.info("Creating substitution network graph")
        
        # Get substitution data
        substitutions = self.error_report['detailed_errors']['substitutions']
        
        if not substitutions:
            logger.warning("No substitution data to create network graph")
            return
        
        # Limit to top substitutions
        substitutions = sorted(substitutions, key=lambda x: x['count'], reverse=True)[:max_edges]
        
        # Create graph
        G = nx.DiGraph()
        
        # Add edges
        for sub in substitutions:
            ref = sub['reference']
            hyp = sub['hypothesis']
            count = sub['count']
            
            G.add_edge(ref, hyp, weight=count, label=str(count))
        
        # Set up the plot
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', alpha=0.8)
        
        # Draw edges with varying width based on weight
        edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        max_width = max(edge_widths)
        normalized_widths = [2 + 5 * (width / max_width) for width in edge_widths]
        
        nx.draw_networkx_edges(G, pos, width=normalized_widths, alpha=0.7, 
                              edge_color='gray', arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Draw edge labels (weights)
        edge_labels = {(u, v): G[u][v]['label'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title('Substitution Error Network\n(Reference → Hypothesis)', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'substitution_network.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_heatmap(self, output_dir: str) -> None:
        """
        Create a heatmap of error patterns.
        
        Args:
            output_dir: Directory to save the heatmap.
        """
        logger.info("Creating error pattern heatmap")
        
        # Get file-level WER data
        file_stats = self.error_report.get('file_stats', {})
        
        if not file_stats:
            logger.warning("No file-level statistics to create heatmap")
            return
        
        # Create dataframe
        data = []
        for file_path, stats in file_stats.items():
            # Extract file name without extension
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            data.append({
                'File': file_name,
                'WER': stats.get('wer', 0),
                'CER': stats.get('cer', 0)
            })
        
        df = pd.DataFrame(data)
        
        # Sort by WER
        df = df.sort_values('WER', ascending=False)
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(df) * 0.3)))
        
        # Create heatmap for WER and CER
        heatmap_data = df.set_index('File')[['WER', 'CER']]
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', 
                   linewidths=.5, cbar_kws={'label': 'Error Rate'})
        
        plt.title('Error Rates by File', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'error_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_by_word_length(self, output_dir: str) -> None:
        """
        Plot error patterns by word length.
        
        Args:
            output_dir: Directory to save the plot.
        """
        logger.info("Plotting error patterns by word length")
        
        # Analyze substitutions by word length
        sub_by_length = defaultdict(int)
        for item in self.error_report['detailed_errors']['substitutions']:
            word_length = len(item['reference'])
            sub_by_length[word_length] += item['count']
        
        # Analyze deletions by word length
        del_by_length = defaultdict(int)
        for item in self.error_report['detailed_errors']['deletions']:
            word_length = len(item['word'])
            del_by_length[word_length] += item['count']
        
        # Analyze insertions by word length
        ins_by_length = defaultdict(int)
        for item in self.error_report['detailed_errors']['insertions']:
            word_length = len(item['word'])
            ins_by_length[word_length] += item['count']
        
        # Combine data
        all_lengths = set(list(sub_by_length.keys()) + list(del_by_length.keys()) + list(ins_by_length.keys()))
        lengths = sorted(all_lengths)
        
        # Create dataframe
        data = []
        for length in lengths:
            data.append({
                'Word Length': length,
                'Substitutions': sub_by_length.get(length, 0),
                'Deletions': del_by_length.get(length, 0),
                'Insertions': ins_by_length.get(length, 0)
            })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 8))
        
        bar_width = 0.25
        index = np.arange(len(lengths))
        
        plt.bar(index, df['Substitutions'], bar_width, label='Substitutions', color='#ff9999')
        plt.bar(index + bar_width, df['Deletions'], bar_width, label='Deletions', color='#66b3ff')
        plt.bar(index + 2*bar_width, df['Insertions'], bar_width, label='Insertions', color='#99ff99')
        
        plt.xlabel('Word Length (characters)', fontsize=12)
        plt.ylabel('Number of Errors', fontsize=12)
        plt.title('Error Patterns by Word Length', fontsize=16)
        plt.xticks(index + bar_width, lengths)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'error_by_word_length.png'), dpi=300, bbox_inches='tight')
        plt.close()
