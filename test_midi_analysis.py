import argparse
import json
from pathlib import Path
from typing import List

from src.analyzer import MIDIAnalyzer, compare_performance
from src.midi_parser import MIDIParser


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_files"


def _list_midi_files(search_root: Path) -> List[Path]:
    exts = {".mid", ".midi"}
    return sorted([p for p in search_root.rglob("*") if p.suffix.lower() in exts])


def run_list(search_root: Path) -> int:
    files = _list_midi_files(search_root)
    if not files:
        print(f"No MIDI files found under: {search_root}")
        return 1

    print(f"Found {len(files)} MIDI files under: {search_root}")
    for i, midi_file in enumerate(files, 1):
        rel = midi_file.relative_to(search_root)
        print(f"{i:>3}. {rel} ({midi_file.stat().st_size} bytes)")
    return 0


def run_solo(midi_path: Path) -> int:
    if not midi_path.exists():
        print(f"File not found: {midi_path}")
        return 2

    parser = MIDIParser()
    analyzer = MIDIAnalyzer()

    parsed_data = parser.parse_midi(str(midi_path))
    if not parsed_data:
        print(f"Failed to parse: {midi_path}")
        return 2

    print(f"Parsed: {midi_path}")
    print(f"Total notes: {len(parsed_data.get('notes', []))}")
    print(f"Duration: {parsed_data.get('total_duration', 0):.2f}s")

    solo_result = analyzer.analyze_solo_performance(str(midi_path))
    analyzer.print_analysis_summary()

    out_file = ROOT / "solo_analysis_results.json"
    out_file.write_text(json.dumps(solo_result, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out_file}")
    return 0


def run_compare(
    reference_path: Path,
    performance_path: Path,
    output_dir: Path,
    alignment_backend: str = "native",
    alignment_model: str = "automatic_hdtw_sym",
) -> int:
    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}")
        return 2
    if not performance_path.exists():
        print(f"Performance file not found: {performance_path}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    result = compare_performance(
        reference_path=str(reference_path),
        performance_path=str(performance_path),
        output_dir=str(output_dir),
        alignment_backend=alignment_backend,
        alignment_model=alignment_model,
    )

    print(f"Analysis complete. Results saved to: {output_dir}")
    metrics = result.get("performance_analysis", {}).get("metrics", {})
    if "performance_score" in metrics:
        score = metrics["performance_score"].get("overall_score", 0)
        grade = metrics["performance_score"].get("grade", "N/A")
        print(f"Performance Score: {score:.1f}%")
        print(f"Grade: {grade}")
    return 0


def run_create_test_midi(output_path: Path) -> int:
    try:
        import pretty_midi
    except ImportError:
        print("pretty_midi is not installed. Run: pip install pretty_midi")
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    start_time = 0.5
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=start_time + i * 0.5,
            end=start_time + i * 0.5 + 0.4,
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(str(output_path))

    print(f"Created: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MIDI analysis smoke-test utility")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List MIDI files in a directory tree")
    p_list.add_argument("--root", type=Path, default=ROOT)

    p_solo = sub.add_parser("solo", help="Run solo analysis on one MIDI file")
    p_solo.add_argument("--midi", type=Path, default=SAMPLE_DIR / "performance.mid")

    p_compare = sub.add_parser("compare", help="Run reference vs performance analysis")
    p_compare.add_argument("--reference", type=Path, default=SAMPLE_DIR / "reference.mid")
    p_compare.add_argument("--performance", type=Path, default=SAMPLE_DIR / "performance.mid")
    p_compare.add_argument("--output", type=Path, default=ROOT / "analysis_results")
    p_compare.add_argument(
        "--alignment-backend",
        type=str,
        default="native",
        choices=["native", "paper_best"],
        help="Choose alignment engine.",
    )
    p_compare.add_argument(
        "--alignment-model",
        type=str,
        default="automatic_hdtw_sym",
        help="Paper backend model (used only when --alignment-backend=paper_best).",
    )

    p_create = sub.add_parser("create-test-midi", help="Create a simple C-major scale MIDI")
    p_create.add_argument("--output", type=Path, default=SAMPLE_DIR / "test_scale.mid")

    args = parser.parse_args()

    if args.command == "list":
        return run_list(args.root.resolve())
    if args.command == "solo":
        return run_solo(args.midi.resolve())
    if args.command == "compare":
        return run_compare(
            args.reference.resolve(),
            args.performance.resolve(),
            args.output.resolve(),
            alignment_backend=args.alignment_backend,
            alignment_model=args.alignment_model,
        )
    if args.command == "create-test-midi":
        return run_create_test_midi(args.output.resolve())

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
