from .midi_parser import MIDIParser

class MIDIAnalyzer:
    def __init__(self):
        """Initialize all analysis components."""
        self.parser = MIDIParser()

    def analyze_solo_performance(self, performance_path):
        """
        Analyze a performance without reference (for practice sessions).
        """
        performance_data = self.parser.parse_midi(performance_path)
        
        # For now, just return the parsed data since phrase_segmenter doesn't exist
        return {
            'parsed_data': performance_data,
            'notes_count': len(performance_data.get('notes', [])),
            'total_duration': performance_data.get('total_duration', 0)
        }
    
    def print_parsed_data(self, performance_path):
        """
        Parse a MIDI file and print the parsed data structure.
        """
        try:
            performance_data = self.parser.parse_midi(performance_path)
            
            print(" MIDI PARSED DATA ")
            print(f"Total notes: {len(performance_data.get('notes', []))}")
            print(f"Total duration: {performance_data.get('total_duration', 0):.2f} seconds")
            print(f"Instruments: {performance_data.get('instruments', [])}")
            
            print("\n FIRST 5 NOTES ")
            notes = performance_data.get('notes', [])
            for i, note in enumerate(notes[:5]):
                print(f"Note {i+1}: pitch={note['pitch']}, start={note['start']:.2f}s, "
                      f"duration={note['duration']:.2f}s, velocity={note['velocity']}")
            
            if len(notes) > 5:
                print(f"... and {len(notes) - 5} more notes")
                
            return performance_data
            
        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return None