# bsb-align

Word-level forced alignment for the **Berean Standard Bible (BSB)** English
text against per-chapter audio recordings (Bob Hays narration from
[openbible.com/audio/hays](https://openbible.com/audio/hays/)).

This repo gives you two ways in:

| Path | Script(s) | When to use |
|---|---|---|
| **Full pipeline** *(recommended)* | `whisper_transcribe.py` → `mms_align_words.py` → `align_words.py` | Maximum-quality timing. Whisper + MMS fusion with header detection, gap-fill, and drift correction. |
| Single-script | `align_book.py` | One-shot MMS-only alignment. GPU-aware, faster, less robust on chapters with intros, music, or narrator repeats. |

Both write the same `output/{BOOK}/{BOOK}_{NNN}_words.json` schema, so you
can mix freely (e.g. fast `align_book.py` pass over the whole Bible, then
`make upgrade` for chapters where playback drifts).

The repo ships pre-aligned timings for **66 books** in `output/`, produced
with `align_book.py`. You only need to re-run the pipeline to upgrade
specific chapters or align books that aren't already there.

---

## Output schema

`output/{BOOK}/{BOOK}_{NNN}_words.json`:

```json
{
  "book": "JON",
  "chapter": "001",
  "verses": {
    "1": [
      {"text": "Now",  "start": 3.90, "end": 4.00, "score": 0.976},
      {"text": "the",  "start": 4.04, "end": 4.12, "score": 0.998},
      {"text": "word", "start": 4.18, "end": 4.38, "score": 0.999}
    ],
    "2": [...]
  }
}
```

The full pipeline additionally writes (only when fusion runs):

| File | Purpose |
|---|---|
| `{BOOK}_{NNN}_words_quality.json` | Per-word `source` (mms vs whisper fallback) + `whisper_score` / `mms_score` diagnostics |
| `{BOOK}_{NNN}_timing.json` | Verse-level start timestamps |
| `{BOOK}_{NNN}.srt` | SRT subtitles |
| `{BOOK}_{NNN}_whisper_words.json`, `_mms_words.json` | Intermediate timelines used by the fusion step |

---

## Setup

Python 3.9+.

```bash
python3 -m venv venv
source venv/bin/activate

# Required (MMS forced alignment + verse export — CUDA, MPS, or CPU)
pip install -r requirements.txt

# Optional but recommended (Whisper transcription — Apple Silicon only)
pip install -r requirements-whisper.txt
```

Model downloads on first run:
- MMS_FA: ~1 GB from torchaudio (cached)
- Whisper large-v3 MLX: ~3 GB from Hugging Face (cached)

`align_book.py` and the full pipeline both auto-detect **CUDA → MPS → CPU**
for MMS_FA. Whisper uses MLX (Apple Silicon GPU only); on other platforms
skip Step 1a and live with MMS-only quality.

---

## Get the audio

Download Bob Hays per-chapter MP3s from
[openbible.com/audio/hays](https://openbible.com/audio/hays/) into a folder
(e.g. `audio/`). The scripts match files by the book code + chapter number
in the filename. Common patterns work out of the box; pass `--audio-glob`
or extend `audio_lookup.BOOK_ALIAS` if your filenames need help.

---

## Recommended usage — the full pipeline

```bash
# 1. Sanity-check on one chapter
make book BOOK=JON CHAPTER=1 AUDIO_DIR=audio/hays

# 2. Inspect output/JON/JON_001_words_quality.json
#    (avg_score should be > 0.5; low_quality_verses should be short)

# 3. Run the rest of the book
make book BOOK=JON AUDIO_DIR=audio/hays

# 4. Upgrade an existing book to fused timing (overwrites _words.json)
make upgrade BOOK=ROM AUDIO_DIR=audio/hays
```

Direct invocation:

```bash
python whisper_transcribe.py --book JON --audio-dir audio/hays
python mms_align_words.py    --book JON --audio-dir audio/hays
python align_words.py        --book JON --audio-dir audio/hays
```

By default each step **skips chapters that already have `_words.json`** —
this protects the committed timings. To overwrite, pass `--force` (or
`make upgrade` / `FORCE=1`).

---

## Upgrading existing timings to the full pipeline

The repo's committed `output/` was produced with `align_book.py` (MMS-only).
To upgrade a chapter or whole book to the higher-quality fused timing:

```bash
make upgrade BOOK=JON AUDIO_DIR=audio/hays              # whole book
make upgrade BOOK=JON CHAPTER=11 AUDIO_DIR=audio/hays   # one chapter
```

`upgrade` is `make book FORCE=1` — it re-runs Whisper transcription, MMS
forced alignment, and the fusion step in order, overwriting the existing
`_words.json` and writing the additional `_words_quality.json` companion.

The output schema is identical, so any consumer of `_words.json` keeps
working unchanged. The fusion adds:

- **Header detection** — the first verse no longer starts at 0.0 s when
  the audio has a spoken book/chapter intro.
- **Per-word fusion** — Whisper falls back in for words MMS scored below
  threshold.
- **Gap-fill / drift correction** — local re-alignment when MMS lands on
  silence or a narrator repeat.
- **Collapse recovery** — restart from a Whisper anchor when MMS loses
  track partway through.

---

## Quick path: `align_book.py`

If you don't have Whisper installed (e.g. Linux/CUDA box) or just want
fast MMS-only timings:

```bash
python align_book.py --book JON --audio-dir audio/hays
python align_book.py --book JON --audio-dir audio/hays --chapter 1
python align_book.py --book GEN --audio-dir audio/hays --force
```

Output schema is identical (`{text, start, end, score}` per word per verse),
so swapping later via `make upgrade` is seamless.

---

## All flags

```
Common to all four scripts:
  --book BOOK              3-letter BSB book code (e.g. JON, GEN, JHN)   [required]
  --audio-dir DIR          Per-chapter mp3 directory                     [required]
  --text-dir DIR           BSB text directory          [default: text/]
  --output-dir DIR         Output directory            [default: output/]
  --chapter N              Process a single chapter
  --audio-glob PATTERN     Custom glob (placeholders: {book}/{book_lc}/{book_title}/{ch}/{ch2}/{ch3})
  --force                  Overwrite existing _words.json
  --dry-run                Show what would be processed and exit

whisper_transcribe.py:
  --model NAME             MLX Whisper model [default: mlx-community/whisper-large-v3-mlx]

mms_align_words.py:
  --redo-collapsed         Re-align only chapters whose intermediate has collapsed regions

align_words.py:
  --redo-no-quality        Re-fuse only chapters that have _words.json but no _words_quality.json
```

---

## Verse-level audio export

After alignment, `export_verses.py` chops the source MP3 into per-verse
clips and writes a `metadata.csv` (with `file_name` and `transcription`
columns, compatible with the `audio_text_tests` pipeline):

```bash
python export_verses.py --json-dir output/JON --audio-dir audio/hays \
    --text-dir text --output-dir dataset/JON
```

---

## Repo layout

```
text/                       BSB text — one file per chapter, one verse per line
                            Filename: {BOOK}_{NNN}_BSB.txt   (e.g. JON_001_BSB.txt)
output/                     Pre-aligned timings (66 books); regenerated/extended by the scripts
audio_lookup.py             Shared find_audio + BOOK_ALIAS map
text_processing.py          Shared text cleaning / normalization
align_book.py               Quick path — MMS-only, GPU-aware, single script
whisper_transcribe.py       Step 1a — Whisper transcription
mms_align_words.py          Step 1b — MMS forced alignment
align_words.py              Step 2  — fusion of MMS + Whisper into final timing
export_verses.py            Chop aligned audio into per-verse clips + metadata.csv
Makefile                    Convenience targets (`make book BOOK=JON`, `make upgrade BOOK=JON`)
requirements.txt            torch, torchaudio, numpy, uroman, torchcodec
requirements-whisper.txt    mlx-whisper                                (Apple Silicon only)
```

---

## Book codes

Standard 3-letter codes used by the BSB filenames. A few common ones:

| Code | Book      |  | Code | Book          |
|------|-----------|--|------|---------------|
| GEN  | Genesis   |  | MAT  | Matthew       |
| EXO  | Exodus    |  | MRK  | Mark          |
| PSA  | Psalms    |  | LUK  | Luke          |
| ISA  | Isaiah    |  | JHN  | John          |
| JON  | Jonah     |  | ACT  | Acts          |
| MAL  | Malachi   |  | ROM  | Romans        |
|      |           |  | REV  | Revelation    |

Run `ls text/ | cut -d_ -f1 | sort -u` for the full list. If Hays uses a
non-standard filename for a book (e.g. `Tts` for Titus), add an entry to
`BOOK_ALIAS` in `audio_lookup.py`.

---

## Troubleshooting

**`WARN no audio for JON 001`** — glob didn't match. Run with `--dry-run`,
inspect filenames, then either pass `--audio-glob` or add an entry to
`audio_lookup.BOOK_ALIAS`.

**Low `avg_score` (< 0.5) on a chapter** — usually means the audio has a
spoken intro that MMS absorbed into the first word. Run the full pipeline
with Whisper available — header detection handles it automatically. Or
trim the leading intro from the audio.

**`mlx-whisper not installed`** — only available on Apple Silicon. Skip
Step 1a; Steps 1b + 2 still produce useful timings (just without header
skip / fallback / drift correction).

**Alignment is slow** — MMS_FA uses CUDA → MPS → CPU; Whisper uses MLX
(Apple Silicon only). On CPU a long chapter (Psalm 119, Isaiah 1, etc.)
can take many minutes per step. This is expected.

**Schema mismatch on an existing chapter** — early-build `_words.json`
files used a different schema. `make upgrade BOOK=...` regenerates them
in the canonical format.

---

## License

Code: MIT — see `LICENSE`.

BSB text: public domain (Berean Standard Bible).
Hays audio: see openbible.com for terms.
