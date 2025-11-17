import pretty_midi

class MIDIParser:
    def __init__(self):
        self.parsed_data = {}
    
    def parse_midi(self, midi_path):
        """Parse MIDI file into structured data."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity,
                        'duration': note.end - note.start
                    })
            
            self.parsed_data = {
                'notes': sorted(notes, key=lambda x: x['start']),
                'total_duration': midi_data.get_end_time(),
                'instruments': [inst.name for inst in midi_data.instruments]
            }
            
            return self.parsed_data
            
        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return {}