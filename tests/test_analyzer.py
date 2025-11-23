#!/usr/bin/env python3
"""
Test the analyzer with a MIDI file
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.analyzer import MIDIAnalyzer

def test_analyzer():
    analyzer = MIDIAnalyzer()
    
    # Replace with the path to your actual MIDI file
    # If you don't have one, create a simple test file first
    midi_file_path = "/home/major/Downloads/example.mid"  # or any MIDI file you have
    
    print("Testing MIDI Analyzer...")
    
    if os.path.exists(midi_file_path):
        print(f"Parsing MIDI file: {midi_file_path}")
        result = analyzer.print_parsed_data(midi_file_path)
        
        if result:
            print("\n SUCCESS: MIDI file parsed successfully!")
        else:
            print("\n FAILED: Could not parse MIDI file")
    else:
        print(f" MIDI file not found: {midi_file_path}")
        print("Please create a sample MIDI file first or use an existing one")

if __name__ == "__main__":
    test_analyzer()