#!/usr/bin/env python3
"""
MMS forced alignment for BSB Bible audio (Step 1b of 2).

Aligns BSB English text against per-chapter audio files (e.g. from
openbible.com/audio/hays/) using torchaudio MMS_FA, producing word-level
timing data for consumption by the fusion script (align_words.py, Step 2).

Output format matches whisper_transcribe.py:
    {
        "book": "JON",
        "chapter": "001",
        "words": [
            {"text": "Now", "start": 1.2, "end": 1.5, "score": 0.85},
            ...
        ]
    }

When a Whisper output exists for the same chapter, it is used to:
  - skip the spoken audio header (book/chapter title, music)
  - recover from MMS collapse (drop-out partway through a chapter)

Usage:
    python mms_align_words.py --book JON --audio-dir audio/hays
    python mms_align_words.py --book JON --audio-dir audio/hays --chapter 1
    python mms_align_words.py --book JON --audio-dir audio/hays --force
    python mms_align_words.py --book JON --audio-dir audio/hays --dry-run

Prerequisites:
    pip install torch torchaudio uroman
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from uroman import Uroman

from text_processing import load_language_config, strip_markers, clean_for_alignment
from align_words import detect_audio_header

# ─── Constants ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEXT_DIR = SCRIPT_DIR / "text"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


# ─── Logging ────────────────────────────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ─── MMS Forced Alignment ──────────────────────────────────────────────────

def load_mms_model():
    """Load MMS_FA model, tokenizer, aligner on CPU and init Uroman."""
    bundle = torchaudio.pipelines.MMS_FA
    log("Loading MMS_FA model on CPU ...")
    t0 = time.time()
    model = bundle.get_model()
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    uroman = Uroman()
    log(f"Model loaded in {time.time() - t0:.1f}s")
    return bundle, model, tokenizer, aligner, uroman


def _prepare_words(text: str, uroman: Uroman, tokenizer) -> Tuple[List[str], List[str]]:
    """Romanize text and filter to tokenizer dictionary.

    Returns (orig_words, clean_rom_words).
    """
    romanized = uroman.romanize_string(text)
    orig_words = text.split()
    rom_words = romanized.split()

    dict_keys = set(tokenizer.dictionary.keys())
    clean_rom_words = []
    for w in rom_words:
        cleaned = "".join(c for c in w if c in dict_keys)
        clean_rom_words.append(cleaned if cleaned else "*")

    return orig_words, clean_rom_words


# Maximum waveform samples to process in one model forward pass.
# ~5 minutes at 16 kHz = 4_800_000 samples. Keeps peak memory manageable
# on machines with limited RAM (e.g. 16 GB MacBook).
_MAX_CHUNK_SAMPLES = 4_800_000

# Overlap between chunks (in samples) to avoid boundary artifacts.
# 0.5 seconds at 16 kHz.
_CHUNK_OVERLAP = 8_000


def _compute_emission_chunked(waveform, model):
    """Run model forward pass, chunking long waveforms to limit memory.

    For waveforms shorter than _MAX_CHUNK_SAMPLES, this is identical to
    a single model(waveform) call. For longer waveforms, processes in
    overlapping chunks and concatenates emissions.
    """
    total_samples = waveform.shape[1]

    if total_samples <= _MAX_CHUNK_SAMPLES:
        with torch.no_grad():
            emission, _ = model(waveform)
        return emission

    emissions = []
    offset = 0
    chunk_idx = 0

    while offset < total_samples:
        end = min(offset + _MAX_CHUNK_SAMPLES, total_samples)
        chunk = waveform[:, offset:end]

        with torch.no_grad():
            chunk_emission, _ = model(chunk)

        if chunk_idx == 0:
            emissions.append(chunk_emission)
        else:
            overlap_samples = min(_CHUNK_OVERLAP, end - offset)
            overlap_ratio = overlap_samples / (end - offset)
            overlap_frames = int(chunk_emission.shape[1] * overlap_ratio)
            emissions.append(chunk_emission[:, overlap_frames:, :])

        if end >= total_samples:
            break

        offset = end - _CHUNK_OVERLAP
        chunk_idx += 1

    return torch.cat(emissions, dim=1)


def _align_waveform(
    waveform,
    text: str,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman: Uroman,
) -> List[dict]:
    """Core MMS_FA alignment on a pre-loaded waveform.

    For long audio (>5 min), the model forward pass is chunked to avoid
    OOM errors while the aligner still operates on the full emission sequence.
    """
    orig_words, clean_rom_words = _prepare_words(text, uroman, tokenizer)

    tokens = tokenizer(clean_rom_words)

    emission = _compute_emission_chunked(waveform, model)

    token_spans = aligner(emission[0], tokens)
    ratio = waveform.shape[1] / emission.shape[1] / bundle.sample_rate

    results = []
    for word_i, word_spans in enumerate(token_spans):
        orig_word = orig_words[word_i] if word_i < len(orig_words) else "?"
        if not word_spans:
            results.append({
                "text": orig_word,
                "start": 0.0,
                "end": 0.0,
                "score": 0.0,
            })
            continue
        start_sec = word_spans[0].start * ratio
        end_sec = word_spans[-1].end * ratio
        score = sum(s.score for s in word_spans) / len(word_spans)

        results.append({
            "text": orig_word,
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "score": round(score, 3),
        })

    return results


def load_audio(audio_path: Path, bundle):
    """Load and resample audio, returning (waveform, sample_rate)."""
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    return waveform, bundle.sample_rate


def run_forced_alignment(
    audio_path: Path,
    text: str,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman: Uroman,
) -> List[dict]:
    """Run MMS_FA forced alignment on full audio (no chunking)."""
    waveform, _ = load_audio(audio_path, bundle)
    return _align_waveform(waveform, text, bundle, model, tokenizer, aligner, uroman)


def realign_from_point(
    waveform,
    sample_rate: int,
    restart_time: float,
    text: str,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman,
    end_time: float = None,
) -> List[dict]:
    """Re-run MMS_FA on audio from restart_time onwards (or to end_time).

    Shared by collapse recovery, gap-fill, and drift correction.
    Slices the waveform, aligns text, adjusts timestamps back to original timeframe.
    """
    start_sample = int(restart_time * sample_rate)
    if end_time is not None:
        end_sample = int(end_time * sample_rate)
        segment = waveform[:, start_sample:end_sample]
    else:
        segment = waveform[:, start_sample:]

    if segment.shape[1] == 0:
        return []

    results = _align_waveform(segment, text, bundle, model, tokenizer, aligner, uroman)

    for r in results:
        r["start"] = round(r["start"] + restart_time, 2)
        r["end"] = round(r["end"] + restart_time, 2)

    return results


def align_segment(
    audio_path: Path,
    text: str,
    start_time: float,
    end_time: float,
    bundle,
    model,
    tokenizer,
    aligner,
    uroman,
) -> List[dict]:
    """Run MMS_FA on a segment of the audio between start_time and end_time."""
    waveform, sample_rate = load_audio(audio_path, bundle)
    return realign_from_point(
        waveform, sample_rate, start_time, text,
        bundle, model, tokenizer, aligner, uroman,
        end_time=end_time,
    )


# ─── Collapse Detection & Restart ─────────────────────────────────────────

COLLAPSE_THRESHOLD = 0.1   # Score at or below this = collapsed
COLLAPSE_MIN_RUN = 5       # Minimum consecutive collapsed words to trigger


def detect_collapse(word_results, threshold=COLLAPSE_THRESHOLD, min_run=COLLAPSE_MIN_RUN):
    """Find first collapse point: min_run consecutive words with score <= threshold."""
    run_start = None
    run_len = 0

    for i, w in enumerate(word_results):
        if w["score"] <= threshold:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= min_run:
                return run_start
        else:
            run_start = None
            run_len = 0

    return None


def _map_word_idx_to_verse(word_idx, cleaned_verses):
    """Map a flat word index to (verse_index, word_offset_in_verse)."""
    pos = 0
    for vi, verse in enumerate(cleaned_verses):
        words = verse.split() if verse else []
        if pos + len(words) > word_idx:
            return vi, word_idx - pos
        pos += len(words)
    return len(cleaned_verses) - 1, 0


def _find_whisper_restart_time(whisper_words, verse_texts, verse_idx, config):
    """Find Whisper's timestamp for the start of a verse."""
    from text_processing import normalize_text

    approx_word_pos = 0
    for vi in range(verse_idx):
        cleaned = clean_for_alignment(verse_texts[vi], config)
        if cleaned:
            approx_word_pos += len(cleaned.split())

    target_verse = clean_for_alignment(verse_texts[verse_idx], config)
    if not target_verse:
        return None
    target_first_word = normalize_text(target_verse.split()[0], config)

    search_start = max(0, approx_word_pos - 10)
    search_end = min(len(whisper_words), approx_word_pos + 30)

    for i in range(search_start, search_end):
        w = whisper_words[i]
        w_norm = normalize_text(w["text"], config)
        if w_norm == target_first_word and w.get("score", 0) > 0.3:
            return w["start"]

    if approx_word_pos < len(whisper_words):
        w = whisper_words[approx_word_pos]
        if w.get("start", 0) > 0:
            return w["start"]

    return None


# ─── File I/O ───────────────────────────────────────────────────────────────

def write_mms_words_json(
    word_results: List[dict], book: str, chapter: str, output_path: Path,
):
    """Write MMS word-level timeline in the same format as whisper_words.json."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "book": book,
        "chapter": chapter,
        "words": [
            {
                "text": w["text"],
                "start": w["start"],
                "end": w["end"],
                "score": w["score"],
            }
            for w in word_results
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def has_null_collapse(mms_path: Path) -> bool:
    """Check if an existing MMS output file has the null-collapse pattern."""
    try:
        with open(mms_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        words = data.get("words", [])
        if len(words) < 4:
            return False
        half = len(words) // 2
        second_half = words[half:]
        bad = sum(1 for w in second_half if w.get("score", 0) <= 0.1)
        return bad / len(second_half) > 0.3
    except Exception:
        return False


def _load_whisper_words(whisper_path: Path) -> Optional[list]:
    """Load Whisper word-level data from a whisper_words.json file."""
    if not whisper_path or not whisper_path.exists():
        return None
    try:
        with open(whisper_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("words", [])
    except (json.JSONDecodeError, OSError):
        return None


# ─── Audio file lookup ────────────────────────────────────────────────────

def find_audio(audio_dir: Path, book: str, chapter: int,
               glob_template: Optional[str]) -> Optional[Path]:
    """Find an audio file for a book/chapter inside audio_dir.

    Tries glob_template first if given (with {book}, {ch2}, {ch3} placeholders),
    then falls back to a few common patterns.
    """
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
    redo_collapsed: bool = False,
) -> List[dict]:
    """Find chapter text+audio pairs for a single BSB book."""
    items = []

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
        out_book_dir = output_dir / book
        mms_path = out_book_dir / f"{book}_{chapter_str}_mms_words.json"

        if mms_path.exists() and not force:
            if redo_collapsed and has_null_collapse(mms_path):
                pass  # include — needs redo
            else:
                continue

        audio_path = find_audio(audio_dir, book, chapter, audio_glob)
        if audio_path is None:
            log(f"  WARN no audio for {book} {chapter_str}")
            continue

        items.append({
            "audio_path": audio_path,
            "text_path": tp,
            "mms_path": mms_path,
            "book": book,
            "chapter": chapter,
            "chapter_str": chapter_str,
        })

    return items


# ─── Chapter Processing ────────────────────────────────────────────────────

def process_chapter(item: dict, bundle, model, tokenizer, aligner, uroman, config,
                    header_skip_time: Optional[float] = None,
                    whisper_path: Optional[Path] = None) -> dict:
    """Align a single chapter using MMS forced alignment.

    If header_skip_time is provided (detected from Whisper), the audio is sliced
    to skip the spoken header (book/chapter title, music) and alignment starts
    from that point. Timestamps are adjusted back to the original timeframe.

    If whisper_path is provided, collapse detection is enabled: when MMS loses
    track partway through, the waveform is sliced using Whisper timestamps and
    MMS is re-run on the remaining audio+text.
    """
    book = item["book"]
    chapter_str = item["chapter_str"]
    audio_path = item["audio_path"]
    text_path = item["text_path"]
    mms_path = item["mms_path"]

    with open(text_path, "r", encoding="utf-8") as f:
        verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]

    while verse_texts and not verse_texts[-1].strip():
        verse_texts.pop()

    cleaned_verses = [clean_for_alignment(v, config) for v in verse_texts]
    non_empty_verses = [v for v in cleaned_verses if v]
    full_text = " ".join(non_empty_verses)
    total_words = len(full_text.split())

    if total_words == 0:
        return {"error": "No words in reference text after cleaning"}

    waveform, sample_rate = load_audio(audio_path, bundle)

    t0 = time.time()
    if header_skip_time and header_skip_time > 0:
        word_results = realign_from_point(
            waveform, sample_rate, header_skip_time, full_text,
            bundle, model, tokenizer, aligner, uroman,
        )
    else:
        word_results = _align_waveform(
            waveform, full_text, bundle, model, tokenizer, aligner, uroman,
        )
    elapsed = time.time() - t0

    # ── Collapse detection & restart ──
    restarted = False
    collapse_idx = detect_collapse(word_results)

    if collapse_idx is not None and whisper_path is not None:
        whisper_words = _load_whisper_words(whisper_path)

        if whisper_words:
            verse_idx, _ = _map_word_idx_to_verse(collapse_idx, non_empty_verses)

            restart_time = _find_whisper_restart_time(
                whisper_words, verse_texts, verse_idx, config,
            )

            if restart_time and restart_time > 0:
                sample_offset = int(restart_time * sample_rate)
                if sample_offset < waveform.shape[1]:
                    remaining_text = " ".join(non_empty_verses[verse_idx:])

                    if remaining_text.strip():
                        t1 = time.time()
                        retry_results = realign_from_point(
                            waveform, sample_rate, restart_time, remaining_text,
                            bundle, model, tokenizer, aligner, uroman,
                        )
                        elapsed += time.time() - t1

                        pre_collapse_word_count = sum(
                            len(v.split()) for v in non_empty_verses[:verse_idx]
                        )
                        word_results = word_results[:pre_collapse_word_count] + retry_results
                        restarted = True

    write_mms_words_json(word_results, book, chapter_str, mms_path)

    scores = [w["score"] for w in word_results if w["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0

    result = {
        "words": total_words,
        "aligned": len(scores),
        "avg_score": round(avg_score, 3),
        "elapsed": round(elapsed, 1),
    }
    if header_skip_time:
        result["header_skipped"] = round(header_skip_time, 1)
    if restarted:
        result["restarted"] = True
        result["restart_verse"] = verse_idx
    return result


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MMS forced alignment for a single BSB book.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--book", required=True,
                        help="3-letter BSB book code (e.g. JON, GEN, JHN)")
    parser.add_argument("--audio-dir", required=True, type=Path,
                        help="Directory containing per-chapter .mp3 files")
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR,
                        help=f"BSB text directory (default: {DEFAULT_TEXT_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--chapter", type=int, default=None,
                        help="Optional single chapter to process")
    parser.add_argument("--audio-glob", type=str, default=None,
                        help="Custom glob pattern with {book}/{book_lc}/{ch}/{ch2}/{ch3} placeholders")
    parser.add_argument("--force", action="store_true",
                        help="Re-align even if output exists")
    parser.add_argument("--redo-collapsed", action="store_true",
                        help="Re-align only chapters whose existing output has collapsed null regions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed")
    args = parser.parse_args()

    book = args.book.upper()

    log("=" * 60)
    log(f"MMS Forced Alignment — {book}")
    log("=" * 60)

    items = discover_work_items(
        book=book,
        audio_dir=args.audio_dir,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        chapter_filter=args.chapter,
        audio_glob=args.audio_glob,
        force=args.force,
        redo_collapsed=args.redo_collapsed,
    )

    if not items:
        log("No chapters to process (all done or no audio+text pairs found)")
        return

    log(f"Found {len(items)} chapter(s) to align")

    if args.dry_run:
        for item in items:
            log(f"  {book} {item['chapter_str']} ← {item['audio_path'].name}")
        return

    bundle, model, tokenizer, aligner, uroman = load_mms_model()
    config = load_language_config("eng")

    processed = 0
    failed = 0

    for idx, item in enumerate(items):
        chapter_str = item["chapter_str"]
        label = f"[{idx + 1}/{len(items)}] {book} {chapter_str}"

        # Look for existing Whisper data for header detection and collapse recovery
        whisper_path = item["mms_path"].parent / f"{book}_{chapter_str}_whisper_words.json"
        if not whisper_path.exists():
            whisper_path = None

        # Detect header from Whisper output
        header_skip_time = None
        if whisper_path:
            whisper_words = _load_whisper_words(whisper_path)
            if whisper_words:
                with open(item["text_path"], "r", encoding="utf-8") as f:
                    verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]
                verse_start, header_text = detect_audio_header(whisper_words, verse_texts, config)
                if verse_start:
                    header_skip_time = verse_start
                    log(f"{label} — header detected ({header_skip_time:.1f}s): \"{header_text}\"")

        try:
            stats = process_chapter(item, bundle, model, tokenizer, aligner, uroman, config,
                                    header_skip_time=header_skip_time,
                                    whisper_path=whisper_path)
            if "error" in stats:
                log(f"{label} — {stats['error']}", "ERROR")
                failed += 1
            else:
                log(f"{label} — {stats['aligned']}/{stats['words']} words, "
                    f"score={stats['avg_score']}, took={stats['elapsed']}s")
                processed += 1
        except KeyboardInterrupt:
            log("Interrupted by user", "WARN")
            break
        except Exception as e:
            log(f"{label} — Failed: {e}", "ERROR")
            failed += 1

    log("")
    log(f"Done: {processed} aligned, {failed} failed")


if __name__ == "__main__":
    main()
