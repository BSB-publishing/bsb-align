# bsb-align

High-quality word-level forced alignment for the **Berean Standard Bible (BSB)**
English text against per-chapter audio recordings.

The pipeline runs three steps:

```
   audio/        text/                              output/{BOOK}/
   ┌──────┐    ┌──────┐                            ┌─────────────────────────────┐
   │ .mp3 │    │ .txt │                            │ {BOOK}_{NNN}_whisper_words  │
   └───┬──┘    └───┬──┘                            │ {BOOK}_{NNN}_mms_words      │
       │           │                               │ {BOOK}_{NNN}_words          │  ← word timings
       │           │     1a: whisper_transcribe.py │ {BOOK}_{NNN}_words_quality  │  ← per-word scores
       │           │     1b: mms_align_words.py    │ {BOOK}_{NNN}_timing         │  ← verse timings
       │           │     2:  align_words.py        │ {BOOK}_{NNN}.srt            │  ← subtitles
       └───────────┴──────────────────────────────►└─────────────────────────────┘
```

When both Whisper and MMS outputs are available, **align_words.py** fuses them:
MMS-FA (which aligns directly against the known reference text) is the timing
backbone, with Whisper used as fallback on low-confidence words. Header
detection skips spoken book/chapter intros, and gap-fill / drift correction
re-runs MMS on local segments where it lost track. The result is consistently
better timing than either source alone.

If you only run **mms_align_words.py** + **align_words.py** (skipping Whisper),
you still get word-level timings — just without header skipping, fallback,
gap-fill, or drift correction.

---

## Repo layout

```
text/                       BSB text — one file per chapter, one verse per line
                            Filename: {BOOK}_{NNN}_BSB.txt   (e.g. JON_001_BSB.txt)
text_processing.py          Shared text cleaning / normalization
whisper_transcribe.py       Step 1a — Whisper transcription (optional, recommended)
mms_align_words.py          Step 1b — MMS forced alignment (always)
align_words.py              Step 2  — fusion of MMS + Whisper into final timing
Makefile                    Convenience targets (`make book BOOK=JON`)
requirements.txt            torch, torchaudio, uroman                 (Step 1b + 2)
requirements-whisper.txt    mlx-whisper                                (Step 1a — Apple Silicon)
output/                     (created on first run) per-book timing JSON, SRT, intermediates
```

The `text/` folder contains the full BSB. You bring the audio.

---

## Setup

Python 3.9+ recommended.

```bash
python3 -m venv venv
source venv/bin/activate

# Required (MMS forced alignment — runs on CPU on any platform)
pip install -r requirements.txt

# Optional but recommended (Whisper transcription — Apple Silicon only)
pip install -r requirements-whisper.txt
```

First run downloads model weights:
- MMS_FA: ~1 GB from torchaudio (cached)
- Whisper large-v3 MLX: ~3 GB from Hugging Face (cached)

Both run on CPU/MLX. Expect somewhere around real-time per chapter on a recent laptop.

---

## Get the audio

The BSB is narrated by **Bob Hays** at openbible.com:

  https://openbible.com/audio/hays/

Download the per-chapter MP3s for the book(s) you want into a directory
(e.g. `audio/`). The pipeline doesn't fetch them — that's on you.

The scripts match audio to chapters by globbing for the chapter number in
the filename. Common patterns work out of the box:

- `JON_001_*.mp3`
- `Jon_01.mp3`
- `BSB_32_Jonah_01_Hays.mp3`

If your filenames don't match, pass `--audio-glob` (see [Custom filename pattern](#custom-filename-pattern)).

---

## Usage

### Quickest path: `make`

```bash
# Whole book — all three steps
make book BOOK=JON AUDIO_DIR=audio/hays

# One chapter
make book BOOK=JON CHAPTER=1 AUDIO_DIR=audio/hays

# Just one step
make whisper BOOK=JON AUDIO_DIR=audio/hays
make mms     BOOK=JON AUDIO_DIR=audio/hays
make align   BOOK=JON AUDIO_DIR=audio/hays
```

`AUDIO_DIR` defaults to `audio/` so you can drop it if your folder is named
that way.

### Direct script invocation

```bash
# Step 1a — Whisper transcription (optional, but improves timing)
python whisper_transcribe.py --book JON --audio-dir audio/hays

# Step 1b — MMS forced alignment (required)
python mms_align_words.py --book JON --audio-dir audio/hays

# Step 2 — fusion (writes the final word + verse timing files)
python align_words.py --book JON --audio-dir audio/hays
```

Each script accepts `--chapter N` to process a single chapter, `--force` to
overwrite existing output, and `--dry-run` to preview matched audio/text pairs.

### Recommended workflow

1. **Pick one chapter and validate first:**
   ```bash
   make book BOOK=JON CHAPTER=1 AUDIO_DIR=audio/hays
   ```
   Inspect `output/JON/JON_001_words_quality.json` — `summary.avg_score`
   should be > 0.5 and `low_quality_verses` should be short. Spot-check a
   few timestamps in `output/JON/JON_001_words.json` against the audio.

2. **Run the whole book:**
   ```bash
   make book BOOK=JON AUDIO_DIR=audio/hays
   ```

3. **Re-process problem chapters** (e.g. those flagged in `_words_quality.json`):
   ```bash
   make book BOOK=JON CHAPTER=11 AUDIO_DIR=audio/hays
   ```
   (`make book` re-runs all three steps; the `--force` flag is implicit only
   for chapters that don't yet have output. To rebuild a chapter that's
   already done, add `--force` to the script invocations.)

### Custom filename pattern

Pass an explicit glob with placeholders if auto-matching misses your files:

```bash
python mms_align_words.py --book JON --audio-dir audio/hays \
    --audio-glob "BSB_*_Jonah_{ch2}_*.mp3"
```

Placeholders: `{book}` (uppercase), `{book_lc}` (lowercase),
`{ch}` (chapter as int), `{ch2}` (zero-padded to 2), `{ch3}` (zero-padded to 3).

---

## Outputs

Per chapter, the pipeline writes to `output/{BOOK}/`:

| File                                  | From   | Purpose                                          |
|---------------------------------------|--------|--------------------------------------------------|
| `{BOOK}_{NNN}_whisper_words.json`     | Step 1a| Raw Whisper word timeline (intermediate)         |
| `{BOOK}_{NNN}.srt`                    | Step 1a| SRT subtitles                                    |
| `{BOOK}_{NNN}_mms_words.json`         | Step 1b| MMS forced-alignment word timeline (intermediate)|
| `{BOOK}_{NNN}_words.json`             | Step 2 | **Final** compact per-verse word timings         |
| `{BOOK}_{NNN}_timing.json`            | Step 2 | **Final** verse-level timestamps                 |
| `{BOOK}_{NNN}_words_quality.json`     | Step 2 | Per-word scores + which source was used          |

`{BOOK}_{NNN}_words.json` is the main artifact:

```json
{
  "book": "JON",
  "chapter": "001",
  "verses": {
    "1":  [0.12, 0.34, 0.51, ...],
    "2":  [4.20, 4.36, 4.50, ...],
    ...
  }
}
```

Each array position corresponds to a word in the cleaned reference text;
the value is the start time in seconds. `null` means the word couldn't be
confidently aligned.

---

## All flags

```
Common to all three scripts:
  --book BOOK              3-letter BSB book code (e.g. JON, GEN, JHN)   [required]
  --audio-dir DIR          Per-chapter mp3 directory  [required for whisper / mms]
  --text-dir DIR           BSB text directory          [default: text/]
  --output-dir DIR         Output directory            [default: output/]
  --chapter N              Process a single chapter
  --audio-glob PATTERN     Custom glob (placeholders: {book}/{book_lc}/{ch}/{ch2}/{ch3})
  --force                  Overwrite existing output
  --dry-run                Show what would be processed and exit

whisper_transcribe.py:
  --model NAME             MLX Whisper model [default: mlx-community/whisper-large-v3-mlx]

mms_align_words.py:
  --redo-collapsed         Re-align only chapters whose output has collapsed regions

align_words.py:
  --audio-dir DIR          (Optional) audio dir — enables gap-fill / drift re-alignment
  --redo-no-quality        Re-fuse only chapters with timing but no quality file
```

---

## Quality features

These all run automatically when both Whisper and MMS outputs are present:

- **Header detection** — skips spoken book/chapter title or music intro before
  verse 1, so the first word's timestamp is correct.
- **Per-word fusion** — MMS-FA timestamps are primary; Whisper is fallback for
  words below the MMS confidence threshold.
- **Gap-fill** — gaps > 3 s between consecutive words trigger a local MMS
  re-alignment to recover the missed timing.
- **Drift correction** — when MMS lands on a narrator's repeated phrase
  instead of the actual verse, the segment is re-aligned from the last good
  anchor.
- **Collapse recovery** — if MMS loses track partway through (run of
  near-zero scores), it restarts from a Whisper-anchored timestamp.
- **Monotonicity enforcement** — within each verse, word timestamps are
  forced non-decreasing.

To enable gap-fill / drift correction during fusion, pass `--audio-dir` to
`align_words.py` — the MMS model is loaded only if needed.

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

Run `ls text/ | cut -d_ -f1 | sort -u` for the full list.

---

## Troubleshooting

**`WARN no audio for JON 001`** — the glob didn't match. Run with `--dry-run`
and try `--audio-glob` with explicit placeholders.

**Low `avg_score` (< 0.5)** — the audio likely has a long spoken intro that
MMS absorbed into the first word. If you have Whisper output, header
detection handles it automatically. If not, run Whisper first
(`make whisper BOOK=JON`), or trim the leading intro from the audio.

**`mlx-whisper not installed`** — only an issue on non-Apple-Silicon machines.
Skip Step 1a; Steps 1b + 2 still produce useful word timings, just without
the quality features that need a Whisper reference.

**Alignment is slow** — MMS_FA runs on CPU, Whisper runs on MLX (Apple
Silicon GPU). A long chapter (Psalm 119, Isaiah 1, etc.) can take many
minutes per step. This is normal.

**Output looks wrong but score is high** — try `--force` to rebuild from
scratch in case an intermediate file is stale.

---

## License

Code: MIT — see `LICENSE`.

BSB text: public domain (Berean Standard Bible).
Hays audio: see openbible.com for terms.
