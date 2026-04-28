#!/usr/bin/env python3
"""
Whisper transcription for BSB Bible audio (Step 1a of 2).

Transcribes per-chapter audio (e.g. from openbible.com/audio/hays/) using
mlx-whisper and saves the raw word-level timeline. Used together with
mms_align_words.py (Step 1b) and align_words.py (Step 2 fusion) to produce
high-quality word timings.

Output (per chapter):
    output/{BOOK}/{BOOK}_{NNN}_whisper_words.json
    output/{BOOK}/{BOOK}_{NNN}.srt

Usage:
    python whisper_transcribe.py --book JON --audio-dir audio/hays
    python whisper_transcribe.py --book JON --audio-dir audio/hays --chapter 1
    python whisper_transcribe.py --book JON --audio-dir audio/hays --force
    python whisper_transcribe.py --book JON --audio-dir audio/hays --dry-run

Prerequisites:
    pip install -r requirements-whisper.txt
    Apple Silicon Mac (for mlx-whisper)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ─── Constants ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEXT_DIR = SCRIPT_DIR / "text"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
WHISPER_LANGUAGE = "en"  # BSB is English


# ─── Logging ────────────────────────────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ─── Whisper Transcription ─────────────────────────────────────────────────

def transcribe_audio(audio_path: Path, model_name: str,
                     language: Optional[str] = WHISPER_LANGUAGE) -> dict:
    """Transcribe a single audio file using mlx-whisper with word timestamps."""
    import mlx_whisper

    kwargs = {
        "path_or_hf_repo": model_name,
        "word_timestamps": True,
    }
    if language:
        kwargs["language"] = language

    return mlx_whisper.transcribe(str(audio_path), **kwargs)


def build_word_timeline(segments: List[dict]) -> List[dict]:
    """Build a flat word-level timeline from Whisper segments.

    Uses word-level timestamps from segments when available; otherwise
    interpolates word positions evenly within each segment.
    """
    timeline = []

    for seg in segments:
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                entry = {
                    "text": w.get("word", w.get("text", "")),
                    "start": w["start"],
                    "end": w.get("end", w["start"]),
                }
                if "probability" in w:
                    entry["score"] = w["probability"]
                timeline.append(entry)
        else:
            text = seg.get("text", "").strip()
            if not text:
                continue
            words = text.split()
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_duration = seg_end - seg_start

            for i, word in enumerate(words):
                if len(words) == 1:
                    t_start = seg_start
                    t_end = seg_end
                else:
                    t_start = seg_start + seg_duration * (i / len(words))
                    t_end = seg_start + seg_duration * ((i + 1) / len(words))
                timeline.append({"text": word, "start": t_start, "end": t_end})

    return timeline


# ─── File I/O ───────────────────────────────────────────────────────────────

def write_whisper_words_json(word_timeline: List[dict], book: str, chapter: str,
                             output_path: Path):
    """Write raw Whisper word-level timeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "book": book,
        "chapter": chapter,
        "words": [
            {
                "text": w["text"],
                "start": round(w["start"], 2),
                "end": round(w.get("end", w["start"]), 2),
                **({"score": round(w["score"], 3)} if "score" in w else {}),
            }
            for w in word_timeline
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: List[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def write_srt(segments: List[dict], output_path: Path):
    srt_content = segments_to_srt(segments)
    if not srt_content:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)


# ─── Audio file lookup ────────────────────────────────────────────────────

def find_audio(audio_dir: Path, book: str, chapter: int,
               glob_template: Optional[str]) -> Optional[Path]:
    book_lc = book.lower()
    ch2 = f"{chapter:02d}"
    ch3 = f"{chapter:03d}"

    patterns = []
    if glob_template:
        patterns.append(glob_template.format(book=book, book_lc=book_lc,
                                             ch2=ch2, ch3=ch3, ch=chapter))
    patterns += [
        f"*{book}*{ch3}*.mp3",
        f"*{book}*{ch2}*.mp3",
        f"*{book_lc}*{ch3}*.mp3",
        f"*{book_lc}*{ch2}*.mp3",
    ]
    for pat in patterns:
        candidates = sorted(audio_dir.glob(pat))
        if candidates:
            return candidates[0]
    return None


# ─── Work Item Discovery ───────────────────────────────────────────────────

def discover_work_items(
    book: str,
    audio_dir: Path,
    text_dir: Path,
    output_dir: Path,
    chapter_filter: Optional[int] = None,
    audio_glob: Optional[str] = None,
    force: bool = False,
) -> List[dict]:
    """Find chapter audio for a single BSB book."""
    items = []
    out_book_dir = output_dir / book

    text_paths = sorted(text_dir.glob(f"{book}_*_BSB.txt"))
    if not text_paths:
        return items

    for tp in text_paths:
        try:
            chapter = int(tp.stem.split("_")[1])
        except (ValueError, IndexError):
            continue

        if chapter_filter is not None and chapter != chapter_filter:
            continue

        chapter_str = f"{chapter:03d}"
        whisper_words_path = out_book_dir / f"{book}_{chapter_str}_whisper_words.json"
        srt_path = out_book_dir / f"{book}_{chapter_str}.srt"

        if whisper_words_path.exists() and not force:
            continue

        audio_path = find_audio(audio_dir, book, chapter, audio_glob)
        if audio_path is None:
            log(f"  WARN no audio for {book} {chapter_str}")
            continue

        items.append({
            "audio_path": audio_path,
            "whisper_words_path": whisper_words_path,
            "srt_path": srt_path,
            "book": book,
            "chapter": chapter,
            "chapter_str": chapter_str,
        })

    return items


# ─── Chapter Processing ────────────────────────────────────────────────────

def process_chapter(chapter: dict, model_name: str) -> dict:
    """Process a single chapter: transcribe audio and write whisper_words + SRT."""
    book = chapter["book"]
    chapter_str = chapter["chapter_str"]
    audio_path = chapter["audio_path"]
    whisper_words_path = chapter["whisper_words_path"]
    srt_path = chapter["srt_path"]

    t0 = time.time()
    result = transcribe_audio(audio_path, model_name)
    transcribe_time = time.time() - t0

    segments = result.get("segments", [])
    duration = segments[-1]["end"] if segments else 0

    word_timeline = build_word_timeline(segments)

    write_whisper_words_json(word_timeline, book, chapter_str, whisper_words_path)
    write_srt(segments, srt_path)

    return {
        "duration": duration,
        "transcribe_time": transcribe_time,
        "words": len(word_timeline),
        "segments": len(segments),
    }


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Whisper transcription for a single BSB book.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--book", required=True,
                        help="3-letter BSB book code (e.g. JON, GEN, JHN)")
    parser.add_argument("--audio-dir", required=True, type=Path,
                        help="Directory containing per-chapter .mp3 files")
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR,
                        help=f"BSB text directory — used for chapter discovery "
                             f"(default: {DEFAULT_TEXT_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--chapter", type=int, default=None,
                        help="Optional single chapter to process")
    parser.add_argument("--audio-glob", type=str, default=None,
                        help="Custom glob pattern with {book}/{book_lc}/{ch}/{ch2}/{ch3} placeholders")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"MLX Whisper model (default: {DEFAULT_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Re-transcribe even if output exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed")
    args = parser.parse_args()

    book = args.book.upper()

    log("=" * 60)
    log(f"Whisper Transcription — {book}")
    log("=" * 60)

    items = discover_work_items(
        book=book,
        audio_dir=args.audio_dir,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        chapter_filter=args.chapter,
        audio_glob=args.audio_glob,
        force=args.force,
    )

    if not items:
        log("No chapters to process (all done or no audio found)")
        return

    log(f"Found {len(items)} chapter(s) to transcribe")

    if args.dry_run:
        for item in items:
            log(f"  {book} {item['chapter_str']} ← {item['audio_path'].name}")
        return

    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        log("mlx-whisper not installed. Run: pip install -r requirements-whisper.txt", "ERROR")
        return

    processed = 0
    failed = 0
    total_audio = 0.0
    total_time = 0.0

    for idx, item in enumerate(items):
        chapter_str = item["chapter_str"]
        label = f"[{idx + 1}/{len(items)}] {book} {chapter_str}"
        log(f"{label} ← {item['audio_path'].name}")

        try:
            stats = process_chapter(item, args.model)
            duration_str = format_duration(stats["duration"])
            speed = stats["duration"] / stats["transcribe_time"] if stats["transcribe_time"] > 0 else 0
            log(f"{label} — {stats['segments']} segments, {stats['words']} words, "
                f"{duration_str} audio, {speed:.1f}x realtime")
            total_audio += stats["duration"]
            total_time += stats["transcribe_time"]
            processed += 1
        except KeyboardInterrupt:
            log("Interrupted by user", "WARN")
            break
        except Exception as e:
            log(f"{label} — Failed: {e}", "ERROR")
            failed += 1

    log("")
    log(f"Done: {processed} transcribed, {failed} failed")
    if total_time > 0:
        log(f"Audio: {format_duration(total_audio)}, "
            f"processing: {format_duration(total_time)}, "
            f"avg {total_audio / total_time:.1f}x realtime")


if __name__ == "__main__":
    main()
