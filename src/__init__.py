"""
MIDI Analysis Module

A comprehensive toolkit for analyzing MIDI performances with educational focus.
Provides tools for parsing, time alignment, phrase segmentation, error analysis,
and generating GPT-ready summaries for piano pedagogy.
"""

__version__ = "0.1.0"


# Import main classes for easy access
from .midi_parser import MIDIParser
from .time_alignment import TimeAlignment
from .phrase_segmentation import PhraseSegmentation
from .error_analysis import ErrorAnalysis
from .json_summarization import JSONSummarization
from .analyzer import MIDIAnalyzer, quick_analyze, compare_performance

# Define what gets imported with "from src import *"
__all__ = [
    'MIDIParser',
    'TimeAlignment', 
    'PhraseSegmentation',
    'ErrorAnalysis',
    'JSONSummarization',
    'MIDIAnalyzer',
    'quick_analyze',
    'compare_performance'
]

# Package metadata
package_info = {
    "name": "midi-analysis-module",
    "version": __version__,
    "description": "Educational MIDI analysis tool for piano pedagogy",
    "modules": [
        "midi_parser",
        "time_alignment", 
        "phrase_segmentation",
        "error_analysis",
        "json_summarization",
        "analyzer"
    ]
}

print(f"Loaded MIDI Analysis Module v{__version__}")