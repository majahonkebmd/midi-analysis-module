# test_midi.py - Test MIDI file access
import os
import glob

maestro_dir = r"C:\Users\majah\Downloads\maestro-v3.0.0-midi\maestro-v3.0.0"

print("Searching for MIDI files in MAESTRO dataset...")
print(f"Directory: {maestro_dir}")

if os.path.exists(maestro_dir):
    # Count MIDI files
    midi_count = 0
    midi_files = []
    
    for root, dirs, files in os.walk(maestro_dir):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_count += 1
                if len(midi_files) < 5:
                    midi_files.append(os.path.join(root, file))
    
    print(f"\nFound {midi_count} MIDI files")
    
    if midi_files:
        print("\nSample files:")
        for i, file in enumerate(midi_files, 1):
            rel_path = os.path.relpath(file, maestro_dir)
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"{i}. {rel_path} ({size_mb:.2f} MB)")
        
        # Test parsing the first file
        print("\n" + "="*60)
        print("Testing MIDI parsing...")
        
        try:
            import pretty_midi
            test_file = midi_files[0]
            print(f"\nParsing: {os.path.basename(test_file)}")
            
            midi_data = pretty_midi.PrettyMIDI(test_file)
            print(f"✓ Successfully parsed")
            print(f"  Duration: {midi_data.get_end_time():.2f} seconds")
            print(f"  Instruments: {len(midi_data.instruments)}")
            print(f"  Total notes: {sum(len(inst.notes) for inst in midi_data.instruments)}")
            
        except Exception as e:
            print(f"✗ Parsing failed: {e}")
            
    else:
        print("\nNo MIDI files found. Check the directory structure.")
else:
    print(f"\nDirectory not found: {maestro_dir}")
    print("\nTry updating the path in the script.")