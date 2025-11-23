import pretty_midi

class MIDIParser:
    def __init__(self):
        self.parsed_data = {}
    
    def parse_piano_only(self, midi_path):
        """Parse MIDI file extracting only piano instruments."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            piano_instruments = []
            
            for instrument in midi_data.instruments:
                # Check if instrument is a piano (Program 0-7 in General MIDI)
                if 0 <= instrument.program <= 7 and not instrument.is_drum:
                    piano_instruments.append(instrument.name)
                    for note in instrument.notes:
                        notes.append({
                            'pitch': note.pitch,
                            'start': note.start,
                            'end': note.end,
                            'velocity': note.velocity,
                            'duration': note.end - note.start,
                            'instrument': instrument.name
                        })
            
            self.parsed_data = {
                'notes': sorted(notes, key=lambda x: x['start']),
                'total_duration': midi_data.get_end_time(),
                'piano_instruments': piano_instruments,
                'piano_note_count': len(notes)
            }
            
            return self.parsed_data
            
        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return {}