"""
Quick Start Guide - Simple examples for getting started.
"""

import os
import sys

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

def safe_import():
    """Safely import the analyzer module."""
    try:
        from src.analyzer import MIDIAnalyzer, quick_analyze, compare_performance
        return MIDIAnalyzer, quick_analyze, compare_performance
    except ImportError:
        try:
            from src.analyzer import MIDIAnalyzer, quick_analyze, compare_performance
            return MIDIAnalyzer, quick_analyze, compare_performance
        except ImportError as e:
            print(f"Import error: {e}")
            print("\nPlease ensure:")
            print("1. You're running from the project root directory")
            print("2. Dependencies are installed: pip install -r requirements.txt")
            print("3. The src/ folder exists with analyzer.py")
            return None, None, None

# Import analyzer
MIDIAnalyzer, quick_analyze, compare_performance = safe_import()

def example1_basic_parsing():
    """Basic example: Parse a MIDI file and show data."""
    if not MIDIAnalyzer:
        return
    
    print("EXAMPLE 1: Basic MIDI Parsing")
    print("-" * 40)
    
    analyzer = MIDIAnalyzer()
    
    # Try to find a MIDI file in common locations
    test_files = [
        "test.mid",
        "sample.mid",
        "example.mid",
        "performance.mid",
        "reference.mid",
        os.path.join("examples", "test.mid"),
        os.path.join("..", "test.mid")
    ]
    
    midi_file = None
    for file in test_files:
        if os.path.exists(file):
            midi_file = file
            break
    
    if midi_file:
        print(f"Found MIDI file: {midi_file}")
        data = analyzer.print_parsed_data(midi_file)
        
        if data:
            print(f"\nParsed {len(data.get('notes', []))} notes")
            print(f"Total duration: {data.get('total_duration', 0):.2f} seconds")
            if data.get('instruments'):
                print(f"Instruments: {', '.join(data.get('instruments', []))}")
    else:
        print("No MIDI file found in common locations.")
        print("Please create a test.mid file or specify a path.")
        
        # Create a test file
        create_test = input("Create a test MIDI file? (y/n): ").strip().lower()
        if create_test == 'y':
            create_test_midi_file("test.mid")
            example1_basic_parsing()  # Try again

def example2_solo_analysis():
    """Analyze a performance without reference."""
    if not MIDIAnalyzer:
        return
    
    print("\n\nEXAMPLE 2: Solo Performance Analysis")
    print("-" * 40)
    
    analyzer = MIDIAnalyzer()
    
    # Check for test files
    test_files = ["test.mid", "performance.mid", "my_performance.mid"]
    performance_file = None
    
    for file in test_files:
        if os.path.exists(file):
            performance_file = file
            break
    
    if performance_file:
        print(f"Analyzing: {performance_file}")
        results = analyzer.analyze_solo_performance(performance_file)
        analyzer.print_analysis_summary()
    else:
        print("No performance file found.")
        print("Please create a test.mid file first or specify a path.")

def example3_compare_with_reference():
    """Compare performance with reference."""
    if not MIDIAnalyzer:
        return
    
    print("\n\nEXAMPLE 3: Reference-Based Analysis")
    print("-" * 40)
    
    analyzer = MIDIAnalyzer()
    
    # Look for test files
    reference_file = "reference.mid" if os.path.exists("reference.mid") else "test.mid"
    performance_file = "performance.mid" if os.path.exists("performance.mid") else "test.mid"
    
    if os.path.exists(reference_file) and os.path.exists(performance_file):
        print(f"Reference: {reference_file}")
        print(f"Performance: {performance_file}")
        
        # Run analysis
        results = analyzer.analyze_with_reference(
            reference_path=reference_file,
            performance_path=performance_file,
            output_dir="quick_analysis"
        )
        
        print("\nAnalysis complete! Check the 'quick_analysis' folder.")
    else:
        print("Test files not found.")
        print("Creating test files for demonstration...")
        
        create_test_midi_file("reference.mid")
        create_test_midi_file("performance.mid")
        
        # Try again
        example3_compare_with_reference()

def example4_using_convenience_functions():
    """Use the convenience functions."""
    if not quick_analyze or not compare_performance:
        return
    
    print("\n\nEXAMPLE 4: Using Convenience Functions")
    print("-" * 40)
    
    # Quick solo analysis
    performance_file = "test.mid"
    if os.path.exists(performance_file):
        print("Running quick analysis...")
        results = quick_analyze(performance_file)
        print(f"Analyzed {len(results.get('parsed_data', {}).get('notes', []))} notes")
    else:
        print(f"File not found: {performance_file}")
    
    # Compare performance
    reference_file = "reference.mid"
    if os.path.exists(reference_file) and os.path.exists(performance_file):
        print("\nRunning comparison...")
        results = compare_performance(reference_file, performance_file, "comparison_results")
        print("Comparison complete!")
    else:
        print("\nMissing files for comparison.")

def create_test_midi_file(filepath):
    """Create a simple test MIDI file for demonstration."""
    try:
        import pretty_midi
        # Create a simple C major scale
        midi = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        # Add a simple C major scale
        notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        for i, pitch in enumerate(notes):
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=i * 0.5,  # Half second apart
                end=i * 0.5 + 0.4
            )
            piano.notes.append(note)
        
        midi.instruments.append(piano)
        midi.write(filepath)
        print(f"  Created test file: {filepath}")
        return True
    except ImportError:
        print("  Could not create test file - pretty_midi not available")
        print("  Install: pip install pretty-midi")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MIDI ANALYSIS MODULE - QUICK START")
    print("=" * 60)
    print("\nThese examples show basic usage.")
    print("They will look for MIDI files or create test files.\n")
    
    # Run examples
    example1_basic_parsing()
    example2_solo_analysis()
    example3_compare_with_reference()
    example4_using_convenience_functions()
    
    print("\n" + "=" * 60)
    print("QUICK START COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Look at demo_usage.py for complete features")
    print("2. Check the analysis_results folder for generated reports")
    print("3. Use the JSON summaries with GPT for feedback")