#!/usr/bin/env python3
"""
Test the analyzer with a MIDI file
"""
import sys
import os

# Add the parent directory to Python path (so src can be found)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analyzer import MIDIAnalyzer

def test_analyzer():
    analyzer = MIDIAnalyzer()
    
    # Test file path - now relative to project root
    midi_file_path = "/home/major/Downloads/example.mid"
    
    print("Testing MIDI Analyzer...")
    
    if os.path.exists(midi_file_path):
        print(f"Parsing MIDI file: {midi_file_path}")
        result = analyzer.print_parsed_data(midi_file_path)
    else:
        print(f"MIDI file not found: {midi_file_path}")

if __name__ == "__main__":
    test_analyzer()