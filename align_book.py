#!/usr/bin/env python3
"""
Word-level forced alignment for a single BSB Bible book.

Aligns BSB English text (text/{BOOK}_{NNN}_BSB.txt — one verse per line)
against per-chapter audio files (e.g. from openbible.com/audio/hays/) using
torchaudio MMS_FA, and writes word-level timing JSON per chapter.

Distilled from example/mms_align_words.py + example/align_words.py:
  - keeps MMS forced alignment with chunked emission for long chapters
  - drops the Whisper/fusion/quality/header/collapse logic
  - drops language-config layer (BSB is English; uses minimal cleaning)
  - one script, one book at a time

Output (per chapter):
    output/{BOOK}/{BOOK}_{NNN}_words.json
    {
      "book": "JON",
      "chapter": "001",
      "verses": {
        "1": [{"text": "Now", "start": 0.12, "end": 0.34, "score": 0.91}, ...],
        ...
      }
    }

Usage:
    python align_book.py --book JON --audio-dir downloads/hays
    python align_book.py --book JON --audio-dir downloads/hays --chapter 1
    python align_book.py --book GEN --audio-dir downloads/hays --force

Audio files are matched by the chapter number embedded in the filename
(e.g. *01*.mp3 or *001*.mp3). Pass --audio-glob to override the pattern.

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


SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEXT_DIR = SCRIPT_DIR / "text"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

_MAX_CHUNK_SAMPLES = 4_800_000  # ~5 min at 16 kHz
_CHUNK_OVERLAP = 8_000          # 0.5 s at 16 kHz


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


# ─── Text ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip punctuation, collapse whitespace. Minimal BSB-English cleaning."""
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def read_chapter_text(text_path: Path) -> Tuple[List[str], str]:
    """Return (cleaned_verses, joined_full_text).

    One verse per line in BSB text files. Empty lines are dropped so the
    verse index in the output matches the line number in the source file
    only when there are no blank lines (BSB files satisfy this).
    """
    with open(text_path, "r", encoding="utf-8") as f:
        verses_raw = [line.rstrip("\n").strip() for line in f]
    verses_raw = [v for v in verses_raw if v]
    cleaned = [clean_text(v) for v in verses_raw]
    full = " ".join(c for c in cleaned if c)
    return cleaned, full


# ─── MMS ───────────────────────────────────────────────────────────────────

def load_mms():
    bundle = torchaudio.pipelines.MMS_FA
    log("Loading MMS_FA model on CPU…")
    t0 = time.time()
    model = bundle.get_model()
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    uroman = Uroman()
    log(f"Model loaded in {time.time() - t0:.1f}s")
    return bundle, model, tokenizer, aligner, uroman


def load_audio(audio_path: Path, bundle):
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, bundle.sample_rate,
        )
    return waveform


def compute_emission(waveform, model):
    """Forward pass, chunked for long audio to keep peak memory low."""
    n = waveform.shape[1]
    if n <= _MAX_CHUNK_SAMPLES:
        with torch.no_grad():
            emission, _ = model(waveform)
        return emission

    emissions = []
    offset = 0
    chunk_idx = 0
    while offset < n:
        end = min(offset + _MAX_CHUNK_SAMPLES, n)
        chunk = waveform[:, offset:end]
        with torch.no_grad():
            chunk_emission, _ = model(chunk)
        if chunk_idx == 0:
            emissions.append(chunk_emission)
        else:
            overlap_samples = min(_CHUNK_OVERLAP, end - offset)
            ratio = overlap_samples / (end - offset)
            overlap_frames = int(chunk_emission.shape[1] * ratio)
            emissions.append(chunk_emission[:, overlap_frames:, :])
        if end >= n:
            break
        offset = end - _CHUNK_OVERLAP
        chunk_idx += 1
    return torch.cat(emissions, dim=1)


def align_words(audio_path: Path, full_text: str,
                bundle, model, tokenizer, aligner, uroman: Uroman) -> List[dict]:
    """Run MMS forced alignment over a whole chapter's audio + text."""
    waveform = load_audio(audio_path, bundle)

    orig_words = full_text.split()
    rom_words = uroman.romanize_string(full_text).split()

    dict_keys = set(tokenizer.dictionary.keys())
    clean_rom = []
    for w in rom_words:
        c = "".join(ch for ch in w if ch in dict_keys)
        clean_rom.append(c if c else "*")

    tokens = tokenizer(clean_rom)
    emission = compute_emission(waveform, model)
    token_spans = aligner(emission[0], tokens)
    ratio = waveform.shape[1] / emission.shape[1] / bundle.sample_rate

    results = []
    for i, spans in enumerate(token_spans):
        word = orig_words[i] if i < len(orig_words) else "?"
        if not spans:
            results.append({"text": word, "start": 0.0, "end": 0.0, "score": 0.0})
            continue
        results.append({
            "text": word,
            "start": round(spans[0].start * ratio, 2),
            "end": round(spans[-1].end * ratio, 2),
            "score": round(sum(s.score for s in spans) / len(spans), 3),
        })
    return results


# ─── Verse mapping ─────────────────────────────────────────────────────────

def map_to_verses(words: List[dict], cleaned_verses: List[str],
                  book: str, chapter_str: str) -> dict:
    """Slice the flat aligned word list back into per-verse buckets.

    MMS aligns 1:1 with the cleaned reference words, so counts per verse
    are sufficient to recover verse boundaries.
    """
    verse_words = {}
    word_idx = 0
    for vi, verse_clean in enumerate(cleaned_verses, start=1):
        n = len(verse_clean.split()) if verse_clean else 0
        verse_words[str(vi)] = words[word_idx:word_idx + n]
        word_idx += n
    return {"book": book, "chapter": chapter_str, "verses": verse_words}


# ─── Audio file lookup ────────────────────────────────────────────────────

def find_audio(audio_dir: Path, book: str, chapter: int,
               glob_template: Optional[str]) -> Optional[Path]:
    """Find an audio file for a book/chapter inside audio_dir.

    Tries glob_template first if given (with {book}, {ch2}, {ch3} placeholders),
    then falls back to a few common patterns.
    """
    candidates: List[Path] = []
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


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Word-align a single BSB book against per-chapter audio.",
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
                        help="Overwrite existing output files")
    parser.add_argument("--dry-run", action="store_true",
                        help="List matched audio/text pairs and exit")
    args = parser.parse_args()

    book = args.book.upper()

    text_paths = sorted(args.text_dir.glob(f"{book}_*_BSB.txt"))
    if not text_paths:
        log(f"No text files for book {book} in {args.text_dir}")
        return

    if args.chapter is not None:
        text_paths = [p for p in text_paths
                      if int(p.stem.split("_")[1]) == args.chapter]
        if not text_paths:
            log(f"Chapter {args.chapter} not found for {book}")
            return

    out_book_dir = args.output_dir / book

    work = []
    for tp in text_paths:
        chapter = int(tp.stem.split("_")[1])
        out = out_book_dir / f"{book}_{chapter:03d}_words.json"
        if out.exists() and not args.force:
            log(f"  skip {book} {chapter:03d} (output exists; use --force)")
            continue
        audio = find_audio(args.audio_dir, book, chapter, args.audio_glob)
        if audio is None:
            log(f"  WARN no audio for {book} {chapter:03d}")
            continue
        work.append((tp, audio, chapter, out))

    if not work:
        log("Nothing to do")
        return

    log(f"{len(work)} chapter(s) to align for {book}")
    if args.dry_run:
        for tp, audio, chapter, out in work:
            log(f"  {book} {chapter:03d}: {audio.name} -> {out}")
        return

    bundle, model, tokenizer, aligner, uroman = load_mms()

    aligned = 0
    failed = 0
    for tp, audio, chapter, out in work:
        chapter_str = f"{chapter:03d}"
        log(f"Aligning {book} {chapter_str} ← {audio.name}")
        try:
            cleaned_verses, full_text = read_chapter_text(tp)
            if not full_text:
                log(f"  empty text after cleaning — skip")
                failed += 1
                continue
            t0 = time.time()
            words = align_words(audio, full_text,
                                bundle, model, tokenizer, aligner, uroman)
            elapsed = time.time() - t0

            payload = map_to_verses(words, cleaned_verses, book, chapter_str)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            scores = [w["score"] for w in words if w["score"] > 0]
            avg = sum(scores) / len(scores) if scores else 0.0
            log(f"  {len(words)} words, avg_score={avg:.2f}, "
                f"took {elapsed:.1f}s → {out}")
            aligned += 1
        except KeyboardInterrupt:
            log("Interrupted")
            break
        except Exception as e:
            log(f"  FAILED: {e}")
            failed += 1

    log(f"Done: {aligned} aligned, {failed} failed")


if __name__ == "__main__":
    main()
