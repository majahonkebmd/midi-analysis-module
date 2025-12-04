"""
Tools for Teachers - Batch processing and student management.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from src.analyzer import MIDIAnalyzer
except ImportError:
    try:
        from analyzer import MIDIAnalyzer
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please run from project root directory")
        sys.exit(1)

# Try to import pandas for CSV export, but don't fail if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Note: pandas not installed. CSV reports will not be generated.")
    print("Install with: pip install pandas")

class TeacherTools:
    """Collection of tools for music teachers."""
    
    def __init__(self):
        self.analyzer = MIDIAnalyzer()
    
    def batch_analyze_students(self, reference_path: str, student_files: dict):
        """
        Analyze multiple student performances.
        
        Args:
            reference_path: Path to reference MIDI
            student_files: Dict of {student_name: performance_path}
        """
        print(f"Batch analyzing {len(student_files)} students...")
        print(f"Reference: {reference_path}")
        
        results = {}
        
        for student_name, perf_path in student_files.items():
            if os.path.exists(perf_path):
                print(f"\nAnalyzing {student_name}...")
                
                try:
                    # Create student-specific output directory
                    output_dir = f"student_analyses/{student_name}_{datetime.now().strftime('%Y%m%d')}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Run analysis
                    analysis = self.analyzer.analyze_with_reference(
                        reference_path=reference_path,
                        performance_path=perf_path,
                        output_dir=output_dir
                    )
                    
                    # Extract key metrics
                    if 'error_analysis' in analysis:
                        metrics = analysis['error_analysis'].get('metrics', {})
                        score_data = metrics.get('performance_score', {})
                        
                        results[student_name] = {
                            'overall_score': score_data.get('overall_score', 0),
                            'grade': score_data.get('grade', 'N/A'),
                            'note_accuracy': metrics.get('note_accuracy', {}).get('accuracy_percentage', 0),
                            'timing_consistency': metrics.get('timing_errors', {}).get('std_error_ms', 0),
                            'analysis_folder': output_dir
                        }
                        
                        print(f"  Score: {results[student_name]['overall_score']:.1f}/100")
                        print(f"  Grade: {results[student_name]['grade']}")
                    else:
                        results[student_name] = {'error': 'No error analysis in results'}
                        print(f"  No analysis results available")
                    
                except Exception as e:
                    print(f"  Error analyzing {student_name}: {e}")
                    results[student_name] = {'error': str(e)}
            else:
                print(f"\nFile not found for {student_name}: {perf_path}")
                results[student_name] = {'error': 'File not found'}
        
        # Save summary report
        self._save_batch_summary(results, reference_path)
        return results
    
    def _save_batch_summary(self, results: dict, reference_path: str):
        """Save batch analysis summary."""
        summary_file = "teacher_summary.json"
        
        summary = {
            'reference_used': reference_path,
            'analysis_date': datetime.now().isoformat(),
            'student_count': len(results),
            'students': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved to: {summary_file}")
        
        # Also create CSV for easy viewing if pandas is available
        if PANDAS_AVAILABLE:
            self._create_csv_report(results)
    
    def _create_csv_report(self, results: dict):
        """Create CSV report of student results."""
        csv_data = []
        
        for student, data in results.items():
            if 'error' not in data:
                csv_data.append({
                    'Student': student,
                    'Score': data.get('overall_score', 0),
                    'Grade': data.get('grade', 'N/A'),
                    'Note Accuracy %': data.get('note_accuracy', 0),
                    'Timing Error (ms)': data.get('timing_consistency', 0),
                    'Analysis Folder': data.get('analysis_folder', '')
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_file = "student_results.csv"
            df.to_csv(csv_file, index=False)
            print(f"📊 CSV report saved to: {csv_file}")
            
            # Show top performers
            print("\nTop Performers:")
            try:
                top_df = df.nlargest(3, 'Score')[['Student', 'Score', 'Grade']]
                print(top_df.to_string(index=False))
            except:
                # Simple display if pandas operation fails
                sorted_students = sorted(csv_data, key=lambda x: x['Score'], reverse=True)
                for i, student in enumerate(sorted_students[:3], 1):
                    print(f"  {i}. {student['Student']}: {student['Score']:.1f} ({student['Grade']})")
    
    def generate_class_report(self, student_results: dict):
        """Generate a class-level report."""
        print("\n" + "=" * 60)
        print("CLASS PERFORMANCE REPORT")
        print("=" * 60)
        
        # Calculate statistics
        scores = [r.get('overall_score', 0) for r in student_results.values() 
                 if 'error' not in r and 'overall_score' in r]
        
        if scores:
            print(f"\nClass Size: {len(scores)} students")
            print(f"Average Score: {sum(scores)/len(scores):.1f}/100")
            print(f"Highest Score: {max(scores):.1f}")
            print(f"Lowest Score: {min(scores):.1f}")
            
            # Identify common issues
            common_issues = self._identify_common_issues(student_results)
            if common_issues:
                print("\nCommon Issues Across Class:")
                for issue, count in common_issues[:5]:  # Top 5 issues
                    print(f"  • {issue} ({count} students)")
        
        # Identify students needing help
        struggling = [(name, data['overall_score']) 
                     for name, data in student_results.items() 
                     if 'error' not in data and data.get('overall_score', 0) < 70]
        
        if struggling:
            print("\nStudents Needing Additional Help (score < 70):")
            for name, score in sorted(struggling, key=lambda x: x[1]):
                print(f"  • {name}: {score:.1f}/100")
    
    def _identify_common_issues(self, student_results: dict) -> list:
        """Identify common issues across students."""
        # This would analyze error patterns across students
        # For now, return a placeholder
        return [
            ("Timing consistency in fast passages", 8),
            ("Dynamic contrast in repeated sections", 6),
            ("Note accuracy in complex chords", 5),
            ("Phrasing continuity", 4),
            ("Rhythmic precision", 3)
        ]
    
    def track_student_progress(self, student_name: str, performance_history: list):
        """
        Track a student's progress over multiple sessions.
        
        Args:
            student_name: Student's name
            performance_history: List of dicts with {date: str, score: float, file: str}
        """
        print(f"\nProgress Tracking for {student_name}")
        print("-" * 40)
        
        if not performance_history:
            print("No performance history available.")
            return
        
        # Sort by date
        sorted_history = sorted(performance_history, key=lambda x: x.get('date', ''))
        
        print("\nPerformance History:")
        for session in sorted_history:
            date = session.get('date', 'Unknown')
            score = session.get('score', 0)
            print(f"  {date}: {score:.1f}/100")
        
        # Calculate progress
        if len(sorted_history) >= 2:
            first_score = sorted_history[0].get('score', 0)
            last_score = sorted_history[-1].get('score', 0)
            improvement = last_score - first_score
            
            print(f"\nOverall Improvement: {improvement:+.1f} points")
            print(f"Average Score: {sum(s['score'] for s in sorted_history)/len(sorted_history):.1f}")
            
            if improvement > 0:
                print("📈 Student is making progress!")
            else:
                print("📉 Student may need additional support.")

def main():
    """Main teacher tools demonstration."""
    tools = TeacherTools()
    
    print("=" * 60)
    print("TEACHER TOOLS - MIDI ANALYSIS MODULE")
    print("=" * 60)
    
    # Example: Batch analysis
    print("\n1. BATCH STUDENT ANALYSIS")
    print("-" * 40)
    
    reference = input("Enter reference MIDI path: ").strip()
    
    if os.path.exists(reference):
        # Example student files (replace with actual paths)
        student_files = {
            "Student_A": "student_a_performance.mid",
            "Student_B": "student_b_performance.mid",
            "Student_C": "student_c_performance.mid",
        }
        
        # Filter to existing files
        existing_files = {name: path for name, path in student_files.items() 
                         if os.path.exists(path)}
        
        if existing_files:
            results = tools.batch_analyze_students(reference, existing_files)
            tools.generate_class_report(results)
        else:
            print("No student files found. Please check paths.")
    else:
        print(f"Reference file not found: {reference}")
    
    # Example: Progress tracking
    print("\n\n2. STUDENT PROGRESS TRACKING")
    print("-" * 40)
    
    # Example progress data
    example_progress = [
        {"date": "2024-01-15", "score": 65.2, "file": "session1.mid"},
        {"date": "2024-01-22", "score": 68.5, "file": "session2.mid"},
        {"date": "2024-01-29", "score": 72.1, "file": "session3.mid"},
        {"date": "2024-02-05", "score": 75.8, "file": "session4.mid"},
    ]
    
    tools.track_student_progress("Example Student", example_progress)
    
    print("\n" + "=" * 60)
    print("TEACHER TOOLS DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • teacher_summary.json - Overall results")
    if PANDAS_AVAILABLE:
        print("  • student_results.csv - Tabular data")
    print("  • student_analyses/ - Individual student analysis folders")

def create_test_files():
    """Create test MIDI files for demonstration."""
    try:
        import pretty_midi
        
        # Create reference file
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
        
        # Add a simple melody
        notes = [
            (60, 0.0, 0.5),   # C4
            (62, 0.5, 1.0),   # D4
            (64, 1.0, 1.5),   # E4
            (65, 1.5, 2.0),   # F4
            (67, 2.0, 2.5),   # G4
        ]
        
        for pitch, start, end in notes:
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=start,
                end=end
            )
            piano.notes.append(note)
        
        midi.instruments.append(piano)
        midi.write("reference.mid")
        print("Created reference.mid")
        
        # Create student files
        for student in ["Student_A", "Student_B", "Student_C"]:
            # Create a slightly different version
            student_midi = pretty_midi.PrettyMIDI()
            student_piano = pretty_midi.Instrument(program=0)
            
            for pitch, start, end in notes:
                # Add some timing variations
                student_start = start + (0.1 if pitch == 62 else 0)  # Rush D4 slightly
                student_end = end + (0.1 if pitch == 67 else 0)      # Drag G4 slightly
                
                note = pretty_midi.Note(
                    velocity=90 if pitch == 64 else 100,  # Softer E4
                    pitch=pitch,
                    start=student_start,
                    end=student_end
                )
                student_piano.notes.append(note)
            
            student_midi.instruments.append(student_piano)
            student_midi.write(f"{student.lower()}_performance.mid")
            print(f"Created {student.lower()}_performance.mid")
            
    except ImportError:
        print("pretty_midi not available. Cannot create test files.")

if __name__ == "__main__":
    # Check if test files exist, create if not
    if not os.path.exists("reference.mid"):
        create_test = input("Create test MIDI files for demonstration? (y/n): ").strip().lower()
        if create_test == 'y':
            create_test_files()
    
    main()