#!/usr/bin/env python3
"""
Word alignment fusion for BSB (Step 2 of 2).

Reads pre-computed word timelines from the output directory:
  - {BOOK}_{NNN}_whisper_words.json  (from whisper_transcribe.py, Step 1a)
  - {BOOK}_{NNN}_mms_words.json      (from mms_align_words.py, Step 1b)

Fuses both sources with the BSB reference text to produce verse-level and
word-level timing files in output/{BOOK}/.

When both sources are available, MMS-FA is used as primary (it aligns
against known text) and Whisper is used as fallback when MMS scores are
below threshold. Header detection, gap-fill re-alignment, and drift
correction tighten timing in problem regions.

When only one source is available, that source is used alone.

Usage:
    python align_words.py --book JON
    python align_words.py --book JON --chapter 1
    python align_words.py --book JON --force
    python align_words.py --book JON --dry-run

Pass --audio-dir to enable gap-fill / drift re-alignment (loads MMS model).
"""

import argparse
import difflib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from text_processing import (
    LanguageConfig,
    clean_for_alignment,
    is_aramaic_chapter,
    load_language_config,
    normalize_text,
    strip_markers,
)

# ─── Constants ──────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEXT_DIR = SCRIPT_DIR / "text"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

WHISPER_MATCH_THRESHOLD = 0.5


# ─── Logging ────────────────────────────────────────────────────────────────

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# ─── File I/O ───────────────────────────────────────────────────────────────

def load_word_timeline(path: Path) -> List[dict]:
    """Load a *_whisper_words.json or *_mms_words.json file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for w in data.get("words", []):
        entry = {
            "text": w["text"],
            "start": w["start"],
            "end": w.get("end", w["start"]),
        }
        if "score" in w:
            entry["score"] = w["score"]
        words.append(entry)
    return words


def write_timing_json(entries: List[dict], output_path: Path):
    """Write verse timing data in the standard format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def write_word_timing_json(word_timing: dict, output_path: Path):
    """Write compact word-level timing data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(word_timing, f, separators=(",", ":"))


def write_quality_json(word_quality: dict, output_path: Path):
    """Write per-word quality data (scores and sources)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(word_quality, f, ensure_ascii=False, indent=2)


# ─── MMS-FA Verse Mapping ──────────────────────────────────────────────────

def _map_mms_to_verses(
    mms_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
    config: LanguageConfig,
) -> Tuple[List[dict], dict, int]:
    """Map flat MMS word results to verse boundaries.

    The MMS words are aligned 1:1 with the cleaned reference text words
    (after strip_markers + clean_for_alignment), so we count words per
    verse and slice accordingly.
    """
    timing_entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}}

    word_idx = 0
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = clean_for_alignment(verse_text, config)

        if not cleaned:
            prev_time = timing_entries[-1]["timestamp"]
            timing_entries.append({
                "book": book,
                "chapter": chapter_str,
                "verse_start": str(verse_num),
                "verse_start_alt": str(verse_num),
                "timestamp": round(prev_time, 2),
            })
            word_timing["verses"][str(verse_num)] = []
            continue

        verse_word_count = len(cleaned.split())
        verse_words = mms_words[word_idx:word_idx + verse_word_count]

        verse_time = None
        for w in verse_words:
            if w["start"] is not None:
                verse_time = w["start"]
                break
        if verse_time is None:
            verse_time = timing_entries[-1]["timestamp"]

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": round(verse_time, 2),
        })

        word_times = []
        for w in verse_words:
            if w["start"] is not None:
                word_times.append(round(w["start"], 2))
            else:
                word_times.append(None)
        word_timing["verses"][str(verse_num)] = word_times

        word_idx += verse_word_count

    return timing_entries, word_timing, len(verse_texts)


# ─── Whisper Verse Alignment ───────────────────────────────────────────────

def _word_similarity(ref_words: List[str], whisper_window: List[str]) -> float:
    """Compute word-level similarity between reference words and a Whisper window."""
    if not ref_words or not whisper_window:
        return 0.0
    matches = 0
    used = set()
    for rw in ref_words:
        best = 0.0
        best_j = -1
        for j, ww in enumerate(whisper_window):
            if j in used:
                continue
            r = difflib.SequenceMatcher(None, rw, ww).ratio()
            if r > best:
                best = r
                best_j = j
        if best >= 0.6 and best_j >= 0:
            matches += 1
            used.add(best_j)
    return matches / len(ref_words)


def _align_whisper_to_verses(
    whisper_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
    config: LanguageConfig,
) -> Tuple[List[dict], dict, int]:
    """Align Whisper word timeline to verse boundaries using fuzzy matching."""
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}}

    total_duration = whisper_words[-1]["end"] if whisper_words else 0
    num_timeline = len(whisper_words)

    norm_timeline = [normalize_text(w["text"], config) for w in whisper_words]

    anchors = {}
    search_from = 0

    for vi, verse_text in enumerate(verse_texts):
        verse_text = verse_text.strip()
        if not verse_text:
            continue
        verse_words = verse_text.split()
        norm_vwords = [normalize_text(w, config) for w in verse_words]
        num_vwords = len(norm_vwords)

        expected_pos = int(num_timeline * vi / len(verse_texts))
        window_start = max(search_from, expected_pos - num_vwords * 3)
        window_end = min(expected_pos + num_vwords * 5, num_timeline)

        best_sim = 0.0
        best_idx = -1

        for i in range(window_start, min(window_end, num_timeline - num_vwords + 1)):
            window = norm_timeline[i:i + num_vwords]
            sim = _word_similarity(norm_vwords, window)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
            if sim > 0.7:
                break

        if best_sim >= WHISPER_MATCH_THRESHOLD and best_idx >= 0:
            anchors[vi] = best_idx
            search_from = best_idx + num_vwords

    matched = len(anchors)

    timing_entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]

    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        verse_text = verse_text.strip()

        if not verse_text:
            prev_time = timing_entries[-1]["timestamp"]
            timing_entries.append({
                "book": book,
                "chapter": chapter_str,
                "verse_start": str(verse_num),
                "verse_start_alt": str(verse_num),
                "timestamp": round(prev_time, 2),
            })
            word_timing["verses"][str(verse_num)] = []
            continue

        verse_words = verse_text.split()
        num_verse_words = len(verse_words)

        if vi in anchors:
            best_idx = anchors[vi]
            timestamp = round(whisper_words[best_idx]["start"], 2)
            word_times = _align_verse_words_whisper(verse_words, whisper_words, best_idx, config)
        else:
            timestamp = _interpolate_verse_time(
                vi, verse_texts, anchors, whisper_words, total_duration,
                timing_entries[-1]["timestamp"],
            )
            word_times = [None] * num_verse_words

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": timestamp,
        })
        word_timing["verses"][str(verse_num)] = word_times

    return timing_entries, word_timing, matched


def _align_verse_words_whisper(
    verse_words: List[str],
    timeline: List[dict],
    timeline_start: int,
    config: LanguageConfig,
) -> list:
    """Align individual verse words against Whisper timeline using fuzzy matching."""
    num_verse_words = len(verse_words)
    result = [None] * num_verse_words
    ti = timeline_start
    max_ti = min(timeline_start + num_verse_words * 3, len(timeline))

    for vi, verse_word in enumerate(verse_words):
        if ti >= max_ti:
            break
        norm_verse = normalize_text(verse_word, config)
        if not norm_verse:
            continue

        best_ratio = 0.0
        best_offset = -1
        look_ahead = min(3, max_ti - ti)
        for offset in range(look_ahead):
            candidate = normalize_text(timeline[ti + offset]["text"], config)
            if not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, norm_verse, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = offset
            if offset > 0:
                merged = normalize_text(
                    " ".join(timeline[ti + j]["text"] for j in range(offset + 1)),
                    config,
                )
                merge_ratio = difflib.SequenceMatcher(None, norm_verse, merged).ratio()
                if merge_ratio > best_ratio:
                    best_ratio = merge_ratio
                    best_offset = 0

        if best_ratio >= 0.4:
            result[vi] = round(timeline[ti + best_offset]["start"], 2)
            ti = ti + best_offset + 1

    return result


def _interpolate_verse_time(
    verse_idx: int,
    verse_texts: List[str],
    anchors: Dict[int, int],
    word_timeline: List[dict],
    total_duration: float,
    prev_timestamp: float,
) -> float:
    """Interpolate a timestamp for an unmatched verse between surrounding anchors."""
    prev_anchor_vi = None
    prev_anchor_time = 0.0
    for vi in range(verse_idx - 1, -1, -1):
        if vi in anchors:
            prev_anchor_vi = vi
            prev_anchor_time = word_timeline[anchors[vi]]["start"]
            break

    next_anchor_vi = None
    next_anchor_time = total_duration
    for vi in range(verse_idx + 1, len(verse_texts)):
        if vi in anchors:
            next_anchor_vi = vi
            next_anchor_time = word_timeline[anchors[vi]]["start"]
            break

    start_vi = (prev_anchor_vi + 1) if prev_anchor_vi is not None else 0
    end_vi = next_anchor_vi if next_anchor_vi is not None else len(verse_texts)

    words_before = 0
    words_total = 0
    for vi in range(start_vi, end_vi):
        vt = verse_texts[vi].strip()
        wc = len(vt.split()) if vt else 0
        if vi < verse_idx:
            words_before += wc
        words_total += wc

    if words_total > 0:
        proportion = words_before / words_total
    else:
        span = end_vi - start_vi
        proportion = (verse_idx - start_vi) / span if span > 0 else 0

    time_range = next_anchor_time - prev_anchor_time
    timestamp = prev_anchor_time + proportion * time_range
    return round(max(timestamp, prev_timestamp), 2)


# ─── Header Detection ─────────────────────────────────────────────────────

def detect_audio_header(
    whisper_words: List[dict],
    verse_texts: List[str],
    config: LanguageConfig,
) -> Tuple[Optional[float], Optional[str]]:
    """Detect spoken audio header (e.g. book/chapter title, music) before verse text.

    Uses two signals:
    1. Gap detection: a silence/music gap > 2s in the first 80s of audio
    2. Text matching: sliding-window match of verse 1 text against Whisper words

    Returns (verse_start_time, header_text) if a header is detected, else (None, None).
    """
    if not whisper_words or not verse_texts:
        return None, None

    first_verse = None
    for vt in verse_texts:
        vt = vt.strip()
        if vt:
            first_verse = vt
            break
    if not first_verse:
        return None, None

    first_verse_words = [normalize_text(w, config) for w in first_verse.split()[:5]]
    first_verse_words = [w for w in first_verse_words if w]
    if not first_verse_words:
        return None, None

    gap_boundary = None
    for j in range(1, len(whisper_words)):
        if whisper_words[j]["start"] > 80.0:
            break
        gap = whisper_words[j]["start"] - whisper_words[j - 1].get("end", whisper_words[j - 1]["start"])
        if gap > 2.0:
            gap_boundary = j
            break

    text_match_idx = None
    search_limit = min(len(whisper_words), 50)
    for i in range(search_limit - len(first_verse_words) + 1):
        window = [normalize_text(whisper_words[i + k]["text"], config) for k in range(len(first_verse_words))]
        matches = 0
        for fvw in first_verse_words:
            best = max(
                (difflib.SequenceMatcher(None, fvw, ww).ratio() for ww in window),
                default=0.0,
            )
            if best >= 0.7:
                matches += 1
        if matches >= max(2, len(first_verse_words) * 0.6):
            text_match_idx = i
            break

    boundary_idx = None
    if gap_boundary is not None and text_match_idx is not None:
        boundary_idx = max(gap_boundary, text_match_idx)
    elif gap_boundary is not None:
        boundary_idx = gap_boundary
    elif text_match_idx is not None and text_match_idx > 0:
        boundary_idx = text_match_idx

    if boundary_idx is None or boundary_idx == 0:
        return None, None

    verse_start_time = whisper_words[boundary_idx]["start"]
    header_text_parts = [w["text"].strip() for w in whisper_words[:boundary_idx]]
    header_text = " ".join(header_text_parts)

    return verse_start_time, header_text


# ─── Per-Word Fusion Logic ─────────────────────────────────────────────────

def _find_whisper_match(
    mms_word: dict,
    whisper_words: List[dict],
    whisper_norm: List[str],
    search_start: int,
    search_end: int,
    config: LanguageConfig,
    max_time_diff: float = 2.0,
) -> Optional[Tuple[int, float]]:
    """Find the best matching Whisper word for an MMS word."""
    mms_start = mms_word["start"]
    mms_norm = normalize_text(mms_word["text"], config)
    if not mms_norm:
        return None

    best_idx = -1
    best_quality = 0.0

    for j in range(search_start, min(search_end, len(whisper_words))):
        w_norm = whisper_norm[j]
        if not w_norm:
            continue

        time_diff = abs(mms_start - whisper_words[j]["start"])
        if time_diff > max_time_diff:
            if whisper_words[j]["start"] > mms_start + max_time_diff:
                break
            continue

        text_sim = difflib.SequenceMatcher(None, mms_norm, w_norm).ratio()
        time_score = 1.0 - time_diff / max_time_diff
        quality = text_sim * 0.7 + time_score * 0.3

        if quality > best_quality:
            best_quality = quality
            best_idx = j

    if best_idx >= 0 and best_quality >= 0.4:
        return (best_idx, best_quality)
    return None


MMS_FALLBACK_THRESHOLD = 0.3  # Default; overridden by config.mms_fallback_threshold


def fuse_words_per_word(
    mms_words: List[dict],
    whisper_words: List[dict],
    verse_texts: List[str],
    book: str,
    chapter_str: str,
    config: LanguageConfig,
    audio_path: Optional[Path] = None,
    mms_components=None,
    gap_fill_dir: Optional[Path] = None,
) -> Tuple[List[dict], dict, dict, dict]:
    """Per-word fusion: MMS primary, Whisper fallback only.

    MMS provides all word timestamps (aligned 1:1 with reference text).
    Whisper is only used as fallback when MMS score < fallback threshold,
    since MMS forced alignment is consistently more accurate for timing.

    If audio_path and mms_components are provided, gaps >3s between
    consecutive MMS words trigger a segment re-alignment. Drift correction
    runs after gap detection to catch cases where MMS landed on a narrator
    repeat instead of the actual verse audio.
    """
    fallback_threshold = config.mms_fallback_threshold
    try:
        ch_num = int(chapter_str)
    except ValueError:
        ch_num = 0
    if is_aramaic_chapter(book, ch_num, config):
        fallback_threshold = 0.01
        log(f"  Aramaic passage detected — using MMS-only (threshold={fallback_threshold})")

    header_end, header_text = detect_audio_header(whisper_words, verse_texts, config)
    if header_end is not None:
        log(f"  Header detected: {header_end:.2f}s "
            f"(Whisper non-matching words before verse text)")

    whisper_norm = [normalize_text(w["text"], config) for w in whisper_words]

    # If MMS was run with header skip, the first verse word has correct timing.
    # If not, the first word may have absorbed the header duration. Detect and fix.
    if header_end is not None and mms_words:
        first = mms_words[0]
        if first["start"] + 0.5 < header_end < first.get("end", first["start"]):
            mms_words = list(mms_words)
            mms_words[0] = {**first, "start": header_end}

    fused_words = []
    whisper_search_start = 0
    words_from_mms = 0
    words_from_whisper = 0

    for mms_w in mms_words:
        mms_score = mms_w.get("score", 0.0)
        fused = {
            "text": mms_w["text"],
            "start": mms_w["start"],
            "end": mms_w.get("end", mms_w["start"]),
            "score": mms_score,
            "source": "mms",
            "whisper_score": None,
        }

        search_end = min(whisper_search_start + 30, len(whisper_words))
        match = _find_whisper_match(
            mms_w, whisper_words, whisper_norm,
            whisper_search_start, search_end, config,
        )

        if match is not None:
            w_idx, match_quality = match
            w_word = whisper_words[w_idx]
            w_score = w_word.get("score", 0.0)
            fused["whisper_score"] = w_score

            if mms_score < fallback_threshold and w_score > mms_score:
                fused["start"] = w_word["start"]
                fused["end"] = w_word.get("end", w_word["start"])
                fused["score"] = w_score
                fused["source"] = "whisper"
                fused["mms_score"] = mms_score
                words_from_whisper += 1
            else:
                words_from_mms += 1

            whisper_search_start = w_idx + 1
        else:
            words_from_mms += 1

        fused_words.append(fused)

    # ── Gap detection and re-alignment ──
    GAP_THRESHOLD = 3.0
    MAX_GAP_FILLS = 3

    gaps_fixed = 0
    attempted_gaps = set()
    for gap_iter in range(MAX_GAP_FILLS):
        gap_found = False
        for i in range(len(fused_words) - 1):
            gap = fused_words[i + 1]["start"] - fused_words[i]["start"]
            gap_key = (i, round(fused_words[i]["start"], 2))
            if gap > GAP_THRESHOLD and fused_words[i]["source"] in ("mms", "mms_gap_fill") and gap_key not in attempted_gaps:
                attempted_gaps.add(gap_key)
                gap_start = fused_words[i]["start"]
                gap_end = fused_words[i + 1]["start"]
                gap_text = fused_words[i]["text"]
                next_text = fused_words[i + 1]["text"]
                log(f"  Gap detected: {gap:.1f}s between "
                    f"'{gap_text}'@{gap_start:.2f}s and "
                    f"'{next_text}'@{gap_end:.2f}s")

                if audio_path and mms_components:
                    whisper_header_end = None
                    if whisper_words:
                        for wi in range(1, min(len(whisper_words), 20)):
                            if whisper_words[wi]["start"] - whisper_words[wi-1]["start"] > 5.0:
                                whisper_header_end = whisper_words[wi-1]["start"] + 0.5
                                break

                    if whisper_header_end and whisper_header_end > gap_start:
                        segment_start = whisper_header_end
                    else:
                        segment_start = max(gap_start + 1.0, gap_end - 2.0)
                    segment_end = gap_end + 1.0
                    segment_text = fused_words[i]["text"]

                    bundle, model, tokenizer, aligner_obj, uroman = mms_components
                    from mms_align_words import align_segment
                    gap_results = align_segment(
                        audio_path, segment_text,
                        segment_start, segment_end,
                        bundle, model, tokenizer, aligner_obj, uroman,
                    )

                    if gap_results and len(gap_results) >= 1:
                        new_start = gap_results[0]["start"]
                        new_score = gap_results[0]["score"]
                        old_start = fused_words[i]["start"]

                        if old_start < new_start < gap_end and new_score > 0.3:
                            log(f"  Gap fixed: '{gap_text}' moved "
                                f"{old_start:.2f}s → {new_start:.2f}s "
                                f"(score={new_score:.2f})")
                            fused_words[i]["start"] = new_start
                            fused_words[i]["end"] = gap_results[0].get("end", new_start)
                            fused_words[i]["score"] = new_score
                            fused_words[i]["source"] = "mms_gap_fill"
                            gaps_fixed += 1
                            gap_found = True

                            if gap_fill_dir:
                                import json as _json
                                gap_fill_dir.mkdir(parents=True, exist_ok=True)
                                gap_file = gap_fill_dir / f"{book}_{chapter_str}_gap_{gap_start:.0f}s.json"
                                with open(gap_file, "w") as gf:
                                    _json.dump({
                                        "original_start": old_start,
                                        "new_start": new_start,
                                        "segment_start": segment_start,
                                        "segment_end": segment_end,
                                        "text": segment_text,
                                        "results": gap_results,
                                        "iteration": gap_iter,
                                    }, gf, indent=2)
                            break
                        else:
                            log(f"  Gap fill rejected: new_start={new_start:.2f}s, "
                                f"score={new_score:.2f}")
                else:
                    log(f"  Gap cannot be fixed (no MMS components available)")
                break

        if not gap_found:
            break

    if gaps_fixed > 0:
        log(f"  Fixed {gaps_fixed} gap(s) via segment re-alignment")

    # ── Gap-triggered MMS drift correction ──
    drift_fixed = 0

    if audio_path and mms_components and whisper_words and gaps_fixed >= 0:
        for i in range(len(fused_words) - 1):
            gap = fused_words[i + 1]["start"] - fused_words[i]["start"]
            if gap < GAP_THRESHOLD:
                continue

            drift_start_idx = None
            last_good_idx = i

            for j in range(i + 1, min(i + 10, len(fused_words))):
                fw = fused_words[j]
                mms_time = fw["start"]
                fw_text_norm = normalize_text(fw["text"], config)
                if not fw_text_norm:
                    continue

                best_wh_time = None
                for ww in whisper_words:
                    if ww["start"] > mms_time - 0.5:
                        break
                    ww_norm = normalize_text(ww["text"], config)
                    if not ww_norm:
                        continue
                    sim = difflib.SequenceMatcher(None, fw_text_norm, ww_norm).ratio()
                    if sim >= 0.6 and ww["score"] > 0.5:
                        time_diff = mms_time - ww["start"]
                        if time_diff > 2.0:
                            best_wh_time = ww["start"]

                if best_wh_time is not None:
                    drift_start_idx = j
                    log(f"  MMS drift near gap: word {j} '{fw['text']}' "
                        f"MMS@{mms_time:.2f}s, Whisper@{best_wh_time:.2f}s "
                        f"(drift={mms_time - best_wh_time:.1f}s)")
                    break

            if drift_start_idx is None:
                continue

            last_good_time = fused_words[last_good_idx]["start"]
            gap_end_time = fused_words[i + 1]["start"]
            restart_time = max(0, last_good_time - 0.5)
            segment_end_time = gap_end_time + 0.5

            gap_word_count = (i + 1) - last_good_idx + 1
            gap_words = fused_words[last_good_idx:last_good_idx + gap_word_count]
            gap_text = " ".join(w["text"] for w in gap_words)

            bundle, model, tokenizer, aligner_obj, uroman_obj = mms_components
            from mms_align_words import realign_from_point, load_audio

            log(f"  Re-running MMS on segment {restart_time:.1f}-{segment_end_time:.1f}s "
                f"for {len(gap_words)} words around gap")

            waveform, sample_rate = load_audio(audio_path, bundle)
            new_results = realign_from_point(
                waveform, sample_rate, restart_time, gap_text,
                bundle, model, tokenizer, aligner_obj, uroman_obj,
                end_time=segment_end_time,
            )

            if new_results and len(new_results) == len(gap_words):
                improved = 0
                for k, (old_fw, new_r) in enumerate(zip(gap_words, new_results)):
                    idx = last_good_idx + k
                    if new_r["score"] > 0.3:
                        old_start = fused_words[idx]["start"]
                        if new_r["start"] < old_start - 0.5:
                            fused_words[idx]["start"] = new_r["start"]
                            fused_words[idx]["end"] = new_r.get("end", new_r["start"])
                            fused_words[idx]["score"] = new_r["score"]
                            fused_words[idx]["source"] = "mms_drift_fix"
                            improved += 1
                        elif abs(new_r["start"] - old_start) < 0.5:
                            if new_r["score"] > fused_words[idx]["score"]:
                                fused_words[idx]["score"] = new_r["score"]

                if improved > 0:
                    drift_fixed += improved
                    log(f"  Drift corrected: {improved}/{len(gap_words)} words moved earlier")

                    if gap_fill_dir:
                        import json as _json
                        gap_fill_dir.mkdir(parents=True, exist_ok=True)
                        drift_file = gap_fill_dir / f"{book}_{chapter_str}_drift_{restart_time:.0f}s.json"
                        with open(drift_file, "w") as df:
                            _json.dump({
                                "last_good_idx": last_good_idx,
                                "drift_start_idx": drift_start_idx,
                                "restart_time": restart_time,
                                "segment_end_time": segment_end_time,
                                "words_corrected": improved,
                                "total_gap_words": len(gap_words),
                            }, df, indent=2)
                else:
                    log(f"  Drift re-alignment: no improvement found")
            elif new_results:
                log(f"  Drift fix: word count mismatch ({len(new_results)} vs {len(gap_words)})")
            else:
                log(f"  Drift fix: no results from re-alignment")

            break

    if drift_fixed > 0:
        log(f"  Fixed {drift_fixed} word(s) via drift correction")

    # Map fused words to verses
    timing_entries = [{
        "book": book,
        "chapter": chapter_str,
        "verse_start": "0",
        "verse_start_alt": "0",
        "timestamp": 0,
    }]
    word_timing = {"book": book, "chapter": chapter_str, "verses": {}}
    quality_verses = {}

    word_idx = 0
    for vi, verse_text in enumerate(verse_texts):
        verse_num = vi + 1
        cleaned = clean_for_alignment(verse_text, config)

        if not cleaned:
            prev_time = timing_entries[-1]["timestamp"]
            timing_entries.append({
                "book": book,
                "chapter": chapter_str,
                "verse_start": str(verse_num),
                "verse_start_alt": str(verse_num),
                "timestamp": round(prev_time, 2),
            })
            word_timing["verses"][str(verse_num)] = []
            quality_verses[str(verse_num)] = []
            continue

        verse_word_count = len(cleaned.split())
        verse_words = fused_words[word_idx:word_idx + verse_word_count]

        verse_time = None
        for w in verse_words:
            if w["start"] is not None:
                verse_time = w["start"]
                break
        if verse_time is None:
            verse_time = timing_entries[-1]["timestamp"]

        timing_entries.append({
            "book": book,
            "chapter": chapter_str,
            "verse_start": str(verse_num),
            "verse_start_alt": str(verse_num),
            "timestamp": round(verse_time, 2),
        })

        word_times = []
        verse_quality = []
        for w in verse_words:
            if w["start"] is not None:
                word_times.append(round(w["start"], 2))
            else:
                word_times.append(None)
            q_entry = {"score": round(w["score"], 3)}
            if w["source"] == "whisper":
                q_entry["source"] = "whisper"
                q_entry["mms_score"] = round(w["mms_score"], 3)
            elif w["whisper_score"] is not None:
                q_entry["whisper_score"] = round(w["whisper_score"], 3)
            verse_quality.append(q_entry)
        word_timing["verses"][str(verse_num)] = word_times
        quality_verses[str(verse_num)] = verse_quality

        word_idx += verse_word_count

    # Enforce monotonicity within each verse
    mono_fixes = 0
    for vnum, times in word_timing["verses"].items():
        for i in range(1, len(times)):
            if times[i] is not None and times[i - 1] is not None:
                if times[i] < times[i - 1]:
                    times[i] = times[i - 1]
                    mono_fixes += 1

    fusion_stats = {
        "total_words": len(fused_words),
        "from_mms": words_from_mms,
        "from_whisper": words_from_whisper,
        "mono_fixes": mono_fixes,
    }

    all_scores = [w["score"] for w in fused_words]
    low_quality_threshold = fallback_threshold
    low_quality_count = sum(1 for s in all_scores if s < low_quality_threshold)
    null_count = sum(
        1 for times in word_timing["verses"].values()
        for t in times if t is None
    )
    low_quality_verses = []
    for vnum, qwords in quality_verses.items():
        if any(w["score"] < low_quality_threshold for w in qwords):
            low_quality_verses.append(vnum)

    word_quality = {
        "book": book,
        "chapter": chapter_str,
        "verses": quality_verses,
        "summary": {
            "total_words": len(fused_words),
            "avg_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
            "low_quality_count": low_quality_count,
            "null_count": null_count,
            "from_whisper": words_from_whisper,
            "from_mms": words_from_mms,
            "low_quality_verses": low_quality_verses,
        },
    }

    return timing_entries, word_timing, word_quality, fusion_stats


# ─── Work Item Discovery ───────────────────────────────────────────────────

def discover_work_items(
    book: str,
    text_dir: Path,
    output_dir: Path,
    audio_dir: Optional[Path] = None,
    audio_glob: Optional[str] = None,
    chapter_filter: Optional[int] = None,
    force: bool = False,
    redo_no_quality: bool = False,
) -> List[dict]:
    """Find chapters that have word timing data (MMS and/or Whisper) for a book."""
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
        mms_path = out_book_dir / f"{book}_{chapter_str}_mms_words.json"
        whisper_path = out_book_dir / f"{book}_{chapter_str}_whisper_words.json"
        timing_path = out_book_dir / f"{book}_{chapter_str}_timing.json"
        words_path = out_book_dir / f"{book}_{chapter_str}_words.json"
        quality_path = out_book_dir / f"{book}_{chapter_str}_words_quality.json"

        has_mms = mms_path.exists()
        has_whisper = whisper_path.exists()

        if not has_mms and not has_whisper:
            continue  # nothing to fuse

        if timing_path.exists() and not force:
            if redo_no_quality:
                if quality_path.exists():
                    continue
            else:
                continue

        # Resolve audio (optional, only needed for gap-fill / drift correction)
        audio_path = None
        if audio_dir is not None:
            from mms_align_words import find_audio
            audio_path = find_audio(audio_dir, book, chapter, audio_glob)

        items.append({
            "mms_path": mms_path if has_mms else None,
            "whisper_path": whisper_path if has_whisper else None,
            "ref_text_path": tp,
            "timing_path": timing_path,
            "words_path": words_path,
            "quality_path": quality_path,
            "audio_path": audio_path,
            "book": book,
            "chapter": chapter,
            "chapter_str": chapter_str,
        })

    return items


# ─── Chapter Processing ────────────────────────────────────────────────────

def process_chapter(item: dict, config: LanguageConfig, mms_components=None) -> dict:
    """Fuse word timelines and produce final verse + word timing.

    Args:
        mms_components: Optional tuple (bundle, model, tokenizer, aligner, uroman)
            for gap-fill re-alignment. If None, gap detection still runs but
            cannot re-align — gaps are logged only.
    """
    book = item["book"]
    chapter_str = item["chapter_str"]
    ref_text_path = item["ref_text_path"]
    timing_path = item["timing_path"]
    words_path = item["words_path"]
    quality_path = item["quality_path"]

    if ref_text_path is None:
        return {"error": "No reference text found"}

    with open(ref_text_path, "r", encoding="utf-8") as f:
        verse_texts = [strip_markers(line.rstrip("\n"), config) for line in f.readlines()]
    while verse_texts and not verse_texts[-1].strip():
        verse_texts.pop()

    verse_count = len(verse_texts)

    mms_words = None
    mms_avg_score = 0.0
    if item["mms_path"]:
        mms_words = load_word_timeline(item["mms_path"])
        if mms_words:
            scores = [w.get("score", 0) for w in mms_words if w.get("score", 0) > 0]
            mms_avg_score = sum(scores) / len(scores) if scores else 0

    whisper_words = None
    whisper_avg_score = 0.0
    if item["whisper_path"]:
        whisper_words = load_word_timeline(item["whisper_path"])
        if whisper_words:
            scores = [w.get("score", 0) for w in whisper_words if w.get("score", 0) > 0]
            whisper_avg_score = sum(scores) / len(scores) if scores else 0

    word_quality = None
    if mms_words and whisper_words:
        audio_path = item.get("audio_path")
        if audio_path and isinstance(audio_path, str):
            audio_path = Path(audio_path)

        gap_fill_dir = item["mms_path"].parent if item.get("mms_path") else None

        final_timing, final_word_timing, word_quality, fusion_stats = fuse_words_per_word(
            mms_words, whisper_words, verse_texts, book, chapter_str, config,
            audio_path=audio_path,
            mms_components=mms_components,
            gap_fill_dir=gap_fill_dir,
        )
        source = "fused"
    elif mms_words:
        final_timing, final_word_timing, _ = _map_mms_to_verses(
            mms_words, verse_texts, book, chapter_str, config,
        )
        source = "mms"
        fusion_stats = None
    elif whisper_words:
        final_timing, final_word_timing, _ = _align_whisper_to_verses(
            whisper_words, verse_texts, book, chapter_str, config,
        )
        source = "whisper"
        fusion_stats = None
    else:
        return {"error": "No usable word timing data"}

    write_timing_json(final_timing, timing_path)
    write_word_timing_json(final_word_timing, words_path)

    if word_quality:
        write_quality_json(word_quality, quality_path)

    result = {
        "verses": verse_count,
        "source": source,
        "mms_score": round(mms_avg_score, 3) if mms_words else None,
        "whisper_score": round(whisper_avg_score, 3) if whisper_words else None,
    }
    if fusion_stats:
        result["fusion"] = fusion_stats
    if word_quality:
        result["quality"] = word_quality["summary"]

    return result


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fuse Whisper + MMS word timelines into verse + word timing for one BSB book.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--book", required=True,
                        help="3-letter BSB book code (e.g. JON, GEN, JHN)")
    parser.add_argument("--text-dir", type=Path, default=DEFAULT_TEXT_DIR,
                        help=f"BSB text directory (default: {DEFAULT_TEXT_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="Audio directory (enables gap-fill / drift re-alignment if provided)")
    parser.add_argument("--audio-glob", type=str, default=None,
                        help="Custom glob pattern with {book}/{book_lc}/{ch}/{ch2}/{ch3} placeholders")
    parser.add_argument("--chapter", type=int, default=None,
                        help="Optional single chapter to process")
    parser.add_argument("--force", action="store_true",
                        help="Re-fuse even if output exists")
    parser.add_argument("--redo-no-quality", action="store_true",
                        help="Re-fuse only chapters that have timing but no quality file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed")
    args = parser.parse_args()

    book = args.book.upper()

    log("=" * 60)
    log(f"Word Alignment Fusion — {book}")
    log("=" * 60)

    config = load_language_config("eng")

    items = discover_work_items(
        book=book,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        audio_dir=args.audio_dir,
        audio_glob=args.audio_glob,
        chapter_filter=args.chapter,
        force=args.force,
        redo_no_quality=args.redo_no_quality,
    )

    if not items:
        log("No chapters to process (no word timing data or all done)")
        return

    log(f"Found {len(items)} chapter(s) to align")

    if args.dry_run:
        for item in items:
            sources = []
            if item["mms_path"]:
                sources.append("MMS")
            if item["whisper_path"]:
                sources.append("Whisper")
            audio_status = "OK" if item["audio_path"] else "—"
            log(f"  {book} {item['chapter_str']} — sources: {'+'.join(sources)}, audio: {audio_status}")
        return

    # Load MMS model only if any chapter has both Whisper + MMS + audio
    # (gap-fill / drift correction requires the model loaded)
    mms_components = None
    needs_mms = any(it["whisper_path"] and it["mms_path"] and it["audio_path"] for it in items)
    if needs_mms:
        from mms_align_words import load_mms_model
        log("Loading MMS_FA model for gap-fill / drift correction…")
        mms_components = load_mms_model()

    processed = 0
    failed = 0

    for idx, item in enumerate(items):
        chapter_str = item["chapter_str"]
        label = f"[{idx + 1}/{len(items)}] {book} {chapter_str}"

        try:
            stats = process_chapter(item, config, mms_components=mms_components)
            if "error" in stats:
                log(f"{label} — {stats['error']}", "ERROR")
                failed += 1
            else:
                parts = [f"{stats['verses']} verses", f"source={stats['source']}"]
                if stats["mms_score"] is not None:
                    parts.append(f"mms_score={stats['mms_score']}")
                if stats.get("whisper_score") is not None:
                    parts.append(f"whisper_score={stats['whisper_score']}")
                if stats.get("fusion"):
                    fs = stats["fusion"]
                    parts.append(
                        f"fusion: {fs['from_whisper']}/{fs['total_words']} from whisper (fallback)"
                    )
                    if fs.get("mono_fixes", 0) > 0:
                        parts.append(f"{fs['mono_fixes']} mono fixes")
                log(f"{label} — {', '.join(parts)}")
                processed += 1
        except Exception as e:
            log(f"{label} — Failed: {e}", "ERROR")
            failed += 1

    log("")
    log(f"Done: {processed} aligned, {failed} failed")


if __name__ == "__main__":
    main()
