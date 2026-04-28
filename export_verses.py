#!/usr/bin/env python3
"""
Export verse-level audio clips and metadata.csv from alignment JSON files.

Takes alignment JSON files produced by align_book.py, chops the source MP3
into verse-level segments using torchaudio, and writes a metadata.csv with
'file_name' and 'transcription' columns (compatible with the audio_text_tests
pipeline).

Usage:
    python export_verses.py --json-dir output/JON --audio-dir downloads/hays --text-dir text --output-dir dataset/JON
    python export_verses.py --json-files output/JON/JON_001_words.json output/JON/JON_002_words.json --audio-dir downloads/hays --text-dir text --output-dir dataset/JON
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio

from audio_lookup import find_audio


def find_text(text_dir: Path, book: str, chapter: int) -> Optional[Path]:
    """Find a text file for a book/chapter inside text_dir."""
    ch_int = int(chapter)
    ch2 = f"{ch_int:02d}"
    ch3 = f"{ch_int:03d}"

    patterns = [
        f"{book}*{chapter}*.txt",
        f"{book}*{ch3}*.txt",
        f"{book}*{ch2}*.txt",
        f"{book}*{ch_int}*.txt",
        f"{book.lower()}*{chapter}*.txt",
        f"{book.lower()}*{ch3}*.txt",
        f"{book.lower()}*{ch2}*.txt",
        f"{book.lower()}*{ch_int}*.txt",
    ]

    for pat in patterns:
        candidates = sorted(text_dir.rglob(pat))
        if candidates:
            return candidates[0]
    return None


# ─── Text loading ──────────────────────────────────────────────────────────

def read_verse_texts(text_path: Path) -> List[str]:
    """Return list of verse strings (one per line), dropping blank lines."""
    with open(text_path, "r", encoding="utf-8") as f:
        verses = [line.rstrip("\n").strip() for line in f]
    return [v for v in verses if v]


# ─── Audio segmentation ────────────────────────────────────────────────────

def segment_verse_audio(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path,
    target_sr: int = 16000,
) -> None:
    """Load audio, slice [start_sec, end_sec), and save as MP3 via torchaudio.

    Writes to a temporary file and moves it into place so the output is never
    partially written.
    """
    waveform, sr = torchaudio.load(str(audio_path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    # Clamp bounds
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(waveform.shape[1], int(end_sec * sr))

    if end_sample <= start_sample:
        raise ValueError(f"Invalid segment: start={start_sec}, end={end_sec}")

    segment = waveform[:, start_sample:end_sample]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(suffix=".mp3", dir=output_path.parent)
    try:
        os.close(fd)
        tmp_path_obj = Path(tmp_path)
        torchaudio.save(str(tmp_path_obj), segment, sr)
        os.replace(str(tmp_path_obj), str(output_path))
    except Exception:
        # Clean up the temp file on failure so it is not mistaken for a complete export
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── JSON parsing ──────────────────────────────────────────────────────────

def parse_alignment_json(json_path: Path) -> Tuple[str, str, dict]:
    """Return (book, chapter, verses_dict) from an alignment JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    book = data.get("book", "")
    chapter = data.get("chapter", "")
    verses = data.get("verses", {})
    return book, chapter, verses


def get_verse_timing(verse_words: List[dict]) -> Tuple[float, float]:
    """Return (start_sec, end_sec) for a verse from its word list."""
    if not verse_words:
        return 0.0, 0.0
    starts = [w["start"] for w in verse_words if w.get("start", 0) > 0]
    ends = [w["end"] for w in verse_words if w.get("end", 0) > 0]
    if not starts or not ends:
        return 0.0, 0.0
    return min(starts), max(ends)


# ─── Main export logic ─────────────────────────────────────────────────────

def export_json(
    json_path: Path,
    audio_dir: Path,
    text_dir: Path,
    output_dir: Path,
    audio_glob: Optional[str],
    target_sr: int = 16000,
) -> Tuple[List[dict], int, int]:
    """Process a single JSON file.

    Returns (csv_rows, exported_count, skipped_count).
    A row is only added to csv_rows when the audio segment is successfully saved
    or was already present from a previous run.
    """
    book, chapter, verses = parse_alignment_json(json_path)
    if not book or not chapter:
        print(f"  WARN: Could not parse book/chapter from {json_path}", file=sys.stderr)
        return [], 0, 0

    ch_int = int(chapter)
    audio_path = find_audio(audio_dir, book, ch_int, audio_glob, recursive=True)
    if audio_path is None:
        print(f"  WARN: No audio found for {book} {chapter}", file=sys.stderr)
        return [], 0, 0

    text_path = find_text(text_dir, book, chapter)
    if text_path is None:
        print(f"  WARN: No text file found for {book} {chapter}", file=sys.stderr)
        return [], 0, 0

    verse_texts = read_verse_texts(text_path)

    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[dict] = []
    exported = 0
    skipped = 0

    for verse_num, verse_words in verses.items():
        v_idx = int(verse_num) - 1
        if v_idx < 0 or v_idx >= len(verse_texts):
            print(f"  WARN: Verse {verse_num} out of range for {book} {chapter}", file=sys.stderr)
            skipped += 1
            continue

        if not verse_words:
            print(f"  WARN: Empty alignment for {book} {chapter}:{verse_num} — skipping", file=sys.stderr)
            skipped += 1
            continue

        start_sec, end_sec = get_verse_timing(verse_words)
        if end_sec <= start_sec:
            print(f"  WARN: Invalid timing for {book} {chapter}:{verse_num} — skipping", file=sys.stderr)
            skipped += 1
            continue

        audio_filename = f"{book}_{chapter}_{int(verse_num):03d}.mp3"
        audio_out_path = audio_out_dir / audio_filename
        rel_path = f"./audio/{audio_filename}"

        if audio_out_path.exists():
            csv_rows.append({
                "file_name": rel_path,
                "transcription": verse_texts[v_idx],
            })
            exported += 1
        else:
            try:
                segment_verse_audio(audio_path, start_sec, end_sec, audio_out_path, target_sr)
                csv_rows.append({
                    "file_name": rel_path,
                    "transcription": verse_texts[v_idx],
                })
                exported += 1
            except Exception as e:
                print(f"  ERROR: Failed to segment {book} {chapter}:{verse_num}: {e}", file=sys.stderr)
                skipped += 1
                continue

    return csv_rows, exported, skipped


def collect_json_paths(inputs: List[Path]) -> List[Path]:
    """Expand a mix of files and directories into a flat list of JSON paths.
    
    Directories are searched recursively for files matching *_words.json.
    """
    paths: List[Path] = []
    for p in inputs:
        if p.is_dir():
            paths.extend(sorted(p.rglob("*_words.json")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"WARN: Path not found: {p}", file=sys.stderr)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Export verse-level audio clips and metadata.csv from alignment JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json-files",
        nargs="+",
        type=Path,
        default=None,
        help="One or more alignment JSON files to process",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Directory containing *_words.json files to process (searched recursively)",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        type=Path,
        help="Directory containing per-chapter .mp3 files",
    )
    parser.add_argument(
        "--text-dir",
        required=True,
        type=Path,
        help="Directory containing BSB text files (e.g., text/JON_001_BSB.txt)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory (will contain audio/ and metadata.csv)",
    )
    parser.add_argument(
        "--audio-glob",
        type=str,
        default=None,
        help="Custom glob pattern with {book}/{book_lc}/{ch}/{ch2}/{ch3} placeholders",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate for output audio (default: 16000)",
    )
    args = parser.parse_args()

    if not args.json_files and not args.json_dir:
        parser.error("Either --json-files or --json-dir is required")

    inputs: List[Path] = []
    if args.json_files:
        inputs.extend(args.json_files)
    if args.json_dir:
        inputs.append(args.json_dir)

    json_paths = collect_json_paths(inputs)
    if not json_paths:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_paths)} JSON file(s) to process")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = args.output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: List[dict] = []
    total_exported = 0
    total_skipped = 0

    for json_path in json_paths:
        print(f"Processing {json_path.name}...")
        rows, exported, skipped = export_json(
            json_path,
            args.audio_dir,
            args.text_dir,
            args.output_dir,
            args.audio_glob,
            args.target_sr,
        )
        csv_rows.extend(rows)
        total_exported += exported
        total_skipped += skipped

    # Write metadata.csv
    csv_path = args.output_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDone: {total_exported} verses exported, {total_skipped} skipped")
    print(f"Audio files: {audio_out_dir}")
    print(f"Metadata CSV: {csv_path}")


if __name__ == "__main__":
    main()
