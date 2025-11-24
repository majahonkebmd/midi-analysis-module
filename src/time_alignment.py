import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pretty_midi
from typing import Dict, List, Tuple, Any
import json

class TimeAlignment:
    def __init__(self, reference_midi: pretty_midi.PrettyMIDI, 
                 performance_midi: pretty_midi.PrettyMIDI):
        """
        Initialize time alignment with reference and performance MIDI files.
        
        Args:
            reference_midi: The reference/humanized MIDI (ground truth)
            performance_midi: The student's performance MIDI to analyze
        """
        self.reference_midi = reference_midi
        self.performance_midi = performance_midi
        self.alignment_path = None
        self.warping_path = None
        
    def extract_note_sequence(self, midi: pretty_midi.PrettyMIDI, 
                            use_velocity: bool = True) -> np.ndarray:
        """
        Extract note sequence for DTW alignment.
        Returns array of [start_time, duration, pitch, velocity] for each note.
        """
        notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Normalize features for better DTW performance
                feature_vector = [
                    note.start,           # Start time in seconds
                    note.end - note.start, # Duration in seconds
                    note.pitch / 127.0,   # Normalized pitch (0-1)
                ]
                if use_velocity:
                    feature_vector.append(note.velocity / 127.0)  # Normalized velocity
                
                notes.append(feature_vector)
        
        # Sort by start time
        notes.sort(key=lambda x: x[0])
        return np.array(notes)
    
    def extract_beat_aligned_features(self, midi: pretty_midi.PrettyMIDI, 
                                    hop_length: float = 0.1) -> np.ndarray:
        """
        Extract beat-aligned features for more robust alignment.
        Creates a time series of musical features at regular intervals.
        """
        # Get beats
        beats = midi.get_beats()
        if len(beats) == 0:
            # Fallback: use regular time intervals if no beats detected
            total_time = midi.get_end_time()
            beats = np.arange(0, total_time, 0.5)  # 500ms intervals
        
        features = []
        for beat_time in beats:
            # Find active notes at this beat
            active_notes = []
            for instrument in midi.instruments:
                for note in instrument.notes:
                    if note.start <= beat_time <= note.end:
                        active_notes.append(note)
            
            # Compute features for this time point
            if active_notes:
                pitches = [n.pitch for n in active_notes]
                velocities = [n.velocity for n in active_notes]
                feature_vector = [
                    np.mean(pitches) / 127.0,      # Average pitch
                    len(active_notes),             # Note density
                    np.mean(velocities) / 127.0,   # Average velocity
                ]
            else:
                # Silence - use zero vector
                feature_vector = [0, 0, 0]
                
            features.append(feature_vector)
        
        return np.array(features)
    
    def compute_dtw_alignment(self, method: str = 'note_sequence', 
                            **kwargs) -> Dict[str, Any]:
        """
        Compute DTW alignment between reference and performance.
        
        Args:
            method: 'note_sequence' or 'beat_features'
            **kwargs: Additional parameters for DTW
            
        Returns:
            Dictionary containing alignment results
        """
        if method == 'note_sequence':
            ref_features = self.extract_note_sequence(self.reference_midi)
            perf_features = self.extract_note_sequence(self.performance_midi)
        else:  # 'beat_features'
            ref_features = self.extract_beat_aligned_features(self.reference_midi)
            perf_features = self.extract_beat_aligned_features(self.performance_midi)
        
        # Compute DTW
        distance, path = fastdtw(ref_features, perf_features, dist=euclidean)
        
        self.alignment_path = path
        self.warping_path = self._refine_warping_path(path)
        
        return {
            'dtw_distance': distance,
            'raw_path': path,
            'refined_path': self.warping_path,
            'reference_features': ref_features.tolist(),
            'performance_features': perf_features.tolist(),
            'alignment_quality': self._compute_alignment_quality(path, ref_features, perf_features)
        }
    
    def _refine_warping_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Refine the DTW path to remove unrealistic warping and ensure monotonicity.
        """
        refined_path = []
        prev_ref, prev_perf = path[0]
        refined_path.append((prev_ref, prev_perf))
        
        for i in range(1, len(path)):
            ref_idx, perf_idx = path[i]
            
            # Ensure monotonic progression (no going backward in time)
            if ref_idx > prev_ref and perf_idx > prev_perf:
                refined_path.append((ref_idx, perf_idx))
                prev_ref, prev_perf = ref_idx, perf_idx
            elif ref_idx == prev_ref and perf_idx > prev_perf:
                # Allow staying in reference while progressing in performance
                refined_path.append((ref_idx, perf_idx))
                prev_perf = perf_idx
            elif ref_idx > prev_ref and perf_idx == prev_perf:
                # Allow staying in performance while progressing in reference
                refined_path.append((ref_idx, perf_idx))
                prev_ref = ref_idx
        
        return refined_path
    
    def _compute_alignment_quality(self, path: List[Tuple[int, int]], 
                                 ref_features: np.ndarray, 
                                 perf_features: np.ndarray) -> Dict[str, float]:
        """Compute alignment quality metrics."""
        distances = []
        for ref_idx, perf_idx in path:
            dist = euclidean(ref_features[ref_idx], perf_features[perf_idx])
            distances.append(dist)
        
        return {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'max_distance': float(np.max(distances)),
            'alignment_length': len(path)
        }
    
    def align_notes(self) -> List[Dict[str, Any]]:
        """
        Align individual notes between reference and performance.
        Returns list of aligned note pairs.
        """
        if self.warping_path is None:
            raise ValueError("Must compute DTW alignment first")
        
        ref_notes = self._get_all_notes_sorted(self.reference_midi)
        perf_notes = self._get_all_notes_sorted(self.performance_midi)
        
        aligned_pairs = []
        
        # For each reference note, find the best matching performance note
        for ref_note in ref_notes:
            best_match = None
            min_distance = float('inf')
            
            for perf_note in perf_notes:
                if not perf_note.get('matched', False):
                    # Calculate distance based on timing and pitch
                    time_dist = abs(ref_note['start'] - perf_note['start'])
                    pitch_dist = abs(ref_note['pitch'] - perf_note['pitch'])
                    
                    # Combined distance metric (weight timing more heavily)
                    distance = 0.7 * time_dist + 0.3 * (pitch_dist / 127.0)
                    
                    if distance < min_distance and time_dist < 2.0:  # 2-second threshold
                        min_distance = distance
                        best_match = perf_note
            
            if best_match:
                best_match['matched'] = True
                aligned_pairs.append({
                    'reference_note': ref_note,
                    'performance_note': best_match,
                    'time_difference': best_match['start'] - ref_note['start'],
                    'pitch_difference': best_match['pitch'] - ref_note['pitch'],
                    'velocity_difference': best_match['velocity'] - ref_note['velocity'],
                    'alignment_confidence': 1.0 / (1.0 + min_distance)  # Convert to confidence score
                })
            else:
                # Reference note has no match (missing note)
                aligned_pairs.append({
                    'reference_note': ref_note,
                    'performance_note': None,
                    'time_difference': None,
                    'pitch_difference': None,
                    'velocity_difference': None,
                    'alignment_confidence': 0.0,
                    'error_type': 'missing_note'
                })
        
        # Find extra notes (performance notes without reference matches)
        extra_notes = [note for note in perf_notes if not note.get('matched', False)]
        for extra_note in extra_notes:
            aligned_pairs.append({
                'reference_note': None,
                'performance_note': extra_note,
                'time_difference': None,
                'pitch_difference': None,
                'velocity_difference': None,
                'alignment_confidence': 0.0,
                'error_type': 'extra_note'
            })
        
        return aligned_pairs
    
    def _get_all_notes_sorted(self, midi: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        """Extract all notes from MIDI and sort by start time."""
        notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append({
                    'start': note.start,
                    'end': note.end,
                    'duration': note.end - note.start,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'instrument': instrument.program if instrument.program else 0,
                    'matched': False  # Track if this note has been matched
                })
        
        # Sort by start time
        notes.sort(key=lambda x: x['start'])
        return notes
    
    def get_alignment_statistics(self, aligned_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive alignment statistics."""
        matched_pairs = [p for p in aligned_pairs if p['performance_note'] is not None and p['reference_note'] is not None]
        missing_notes = [p for p in aligned_pairs if p['performance_note'] is None]
        extra_notes = [p for p in aligned_pairs if p['reference_note'] is None]
        
        time_differences = [p['time_difference'] for p in matched_pairs]
        pitch_differences = [p['pitch_difference'] for p in matched_pairs]
        velocity_differences = [p['velocity_difference'] for p in matched_pairs]
        confidence_scores = [p['alignment_confidence'] for p in matched_pairs]
        
        return {
            'alignment_summary': {
                'total_reference_notes': len([p for p in aligned_pairs if p['reference_note'] is not None]),
                'total_performance_notes': len([p for p in aligned_pairs if p['performance_note'] is not None]),
                'successfully_aligned': len(matched_pairs),
                'missing_notes': len(missing_notes),
                'extra_notes': len(extra_notes),
                'alignment_rate': len(matched_pairs) / len([p for p in aligned_pairs if p['reference_note'] is not None])
            },
            'timing_analysis': {
                'mean_time_difference': float(np.mean(time_differences)) if time_differences else 0,
                'std_time_difference': float(np.std(time_differences)) if time_differences else 0,
                'max_time_difference': float(np.max(np.abs(time_differences))) if time_differences else 0,
                'rushing_tendency': len([td for td in time_differences if td < -0.1]) / len(time_differences) if time_differences else 0,
                'dragging_tendency': len([td for td in time_differences if td > 0.1]) / len(time_differences) if time_differences else 0
            },
            'pitch_analysis': {
                'pitch_errors': len([pd for pd in pitch_differences if pd != 0]),
                'mean_pitch_difference': float(np.mean(pitch_differences)) if pitch_differences else 0
            },
            'confidence_metrics': {
                'mean_confidence': float(np.mean(confidence_scores)) if confidence_scores else 0,
                'low_confidence_alignments': len([c for c in confidence_scores if c < 0.5])
            }
        }
    
    def export_alignment_report(self, output_path: str):
        """Export comprehensive alignment report as JSON."""
        alignment_result = self.compute_dtw_alignment()
        aligned_notes = self.align_notes()
        statistics = self.get_alignment_statistics(aligned_notes)
        
        report = {
            'alignment_method': 'dtw_note_sequence',
            'timestamp': np.datetime64('now').astype(str),
            'alignment_metrics': alignment_result['alignment_quality'],
            'note_alignment': aligned_notes,
            'statistics': statistics,
            'warping_path_sample': self.warping_path[:100] if self.warping_path else []  # Sample for size
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

# Utility function for easy usage
def align_midi_files(reference_path: str, performance_path: str, 
                    output_report: str = None) -> Dict[str, Any]:
    """
    Convenience function to align two MIDI files and optionally save report.
    
    Args:
        reference_path: Path to reference MIDI file
        performance_path: Path to performance MIDI file
        output_report: Optional path to save alignment report
        
    Returns:
        Alignment results dictionary
    """
    reference_midi = pretty_midi.PrettyMIDI(reference_path)
    performance_midi = pretty_midi.PrettyMIDI(performance_path)
    
    aligner = TimeAlignment(reference_midi, performance_midi)
    report = aligner.export_alignment_report(output_report or 'alignment_report.json')
    
    return report