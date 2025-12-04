"""
Demo Usage of MIDI Analysis Module - IMPROVED VERSION
"""

import os
import sys
import glob

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# Now import
try:
    from src.analyzer import MIDIAnalyzer
    print("✓ Imported from src.analyzer")
except ImportError as e:
    print(f"Import error: {e}")
    print("\nMake sure you:")
    print("1. Run from project root directory")
    print("2. Have installed dependencies: pip install -r requirements.txt")
    sys.exit(1)

def find_midi_files(directory):
    """Find MIDI files in a directory."""
    midi_files = []
    patterns = ['*.mid', '*.midi', '*.MID', '*.MIDI']
    
    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        files = glob.glob(full_pattern)
        midi_files.extend(files)
    
    return sorted(midi_files)

def select_midi_file(directory, file_type="reference"):
    """Let user select a MIDI file from a directory."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None
    
    if os.path.isfile(directory) and directory.lower().endswith(('.mid', '.midi')):
        # It's already a MIDI file
        return directory
    
    # It's a directory, find MIDI files
    midi_files = find_midi_files(directory)
    
    if not midi_files:
        print(f"No MIDI files found in: {directory}")
        
        # Search recursively
        print("Searching recursively...")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.mid', '.midi')):
                    filepath = os.path.join(root, file)
                    midi_files.append(filepath)
        
        if not midi_files:
            print("Still no MIDI files found.")
            return None
    
    print(f"\nFound {len(midi_files)} MIDI files in {directory}")
    print("\nSelect a file:")
    for i, file in enumerate(midi_files[:10], 1):  # Show first 10
        filename = os.path.basename(file)
        print(f"  {i}. {filename}")
    
    if len(midi_files) > 10:
        print(f"  ... and {len(midi_files) - 10} more files")
    
    try:
        choice = int(input(f"\nEnter selection (1-{min(10, len(midi_files))}): "))
        if 1 <= choice <= len(midi_files):
            return midi_files[choice - 1]
        else:
            print("Invalid selection")
            return midi_files[0] if midi_files else None
    except:
        # Return first file by default
        return midi_files[0] if midi_files else None

def main():
    """
    Main demonstration of the MIDI analysis pipeline.
    """
    print("=" * 60)
    print("MIDI ANALYSIS MODULE - DEMONSTRATION")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = MIDIAnalyzer()
    
    # Default MAESTRO dataset location
    maestro_dir = r"C:\Users\majah\Downloads\maestro-v3.0.0-midi\maestro-v3.0.0"
    
    print("\nLooking for MAESTRO dataset...")
    if os.path.exists(maestro_dir):
        print(f"Found MAESTRO dataset at: {maestro_dir}")
        
        # List subdirectories (years)
        years = [d for d in os.listdir(maestro_dir) 
                if os.path.isdir(os.path.join(maestro_dir, d))]
        
        if years:
            print("\nAvailable years:")
            for year in sorted(years)[:10]:  # Show first 10 years
                print(f"  • {year}")
            
            # Select a year
            year_choice = input("\nEnter a year (or press Enter for 2009): ").strip()
            if not year_choice:
                year_choice = "2009"
            
            year_dir = os.path.join(maestro_dir, year_choice)
            
            if os.path.exists(year_dir):
                # Select reference file
                print(f"\nSelecting reference MIDI from {year_choice}...")
                reference_path = select_midi_file(year_dir, "reference")
                
                # Select performance file (could be same or different)
                print(f"\nSelecting performance MIDI from {year_choice}...")
                performance_path = select_midi_file(year_dir, "performance")
                
                if reference_path and performance_path:
                    print(f"\n✓ Reference: {os.path.basename(reference_path)}")
                    print(f"✓ Performance: {os.path.basename(performance_path)}")
                else:
                    print("\n⚠️ Could not find MIDI files. Using test files instead.")
                    reference_path = "reference.mid"
                    performance_path = "performance.mid"
            else:
                print(f"Year directory not found: {year_dir}")
                reference_path = "reference.mid"
                performance_path = "performance.mid"
        else:
            print("No year directories found in MAESTRO dataset.")
            reference_path = "reference.mid"
            performance_path = "performance.mid"
    else:
        print("MAESTRO dataset not found at default location.")
        print("Please provide paths manually.")
        reference_path = input("\nEnter path to reference MIDI file: ").strip()
        performance_path = input("Enter path to performance MIDI file: ").strip()
    
    # Ensure we have file paths
    if not reference_path or not performance_path:
        print("\n⚠️ No file paths provided. Creating test files...")
        reference_path = "reference.mid"
        performance_path = "performance.mid"
    
    print("\n" + "=" * 60)
    print("ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Test parsing first
    print("\n1. TEST PARSING")
    print("-" * 40)
    
    for path, name in [(reference_path, "Reference"), (performance_path, "Performance")]:
        print(f"\nParsing {name}: {os.path.basename(path)}")
        if os.path.exists(path):
            try:
                parsed_data = analyzer.print_parsed_data(path)
                if parsed_data:
                    print(f"✓ {name} parsed successfully")
                else:
                    print(f"✗ {name} parsing failed")
            except Exception as e:
                print(f"✗ Error parsing {name}: {e}")
        else:
            print(f"✗ File not found: {path}")
    
    # Continue with analysis
    print("\n\n2. FULL ANALYSIS")
    print("-" * 40)
    
    # Ask for output directory
    output_dir = input("\nEnter output directory (press Enter for 'analysis_results'): ").strip()
    if not output_dir:
        output_dir = "analysis_results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("\nRunning complete analysis pipeline...")
        
        # Run complete analysis
        results = analyzer.analyze_with_reference(
            reference_path=reference_path,
            performance_path=performance_path,
            output_dir=output_dir
        )
        
        print("\n✅ Analysis Complete!")
        print(f"Reports saved to: {os.path.abspath(output_dir)}")
        
        # Show summary
        analyzer.print_analysis_summary()
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()