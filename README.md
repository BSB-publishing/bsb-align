# bsb-align

Word-level forced alignment of the **Berean Standard Bible (BSB)** English text against
per-chapter audio recordings, using `torchaudio`'s MMS_FA model.

One small script (`align_book.py`) takes a book code (e.g. `JON`) and an audio
directory, and produces per-verse word timings as JSON.

```
output/JON/JON_001_words.json
{
  "book": "JON",
  "chapter": "001",
  "verses": {
    "1": [
      {"text": "Now",  "start": 0.12, "end": 0.34, "score": 0.91},
      {"text": "the",  "start": 0.36, "end": 0.42, "score": 0.88},
      ...
    ],
    ...
  }
}
```

This is a stripped-down, single-script alternative to a fuller multi-source
pipeline (Whisper + MMS fusion, quality scoring, header detection, drift
correction, language configs, etc.). Use this repo when you just want word
timings for one English book and nothing more.

---

## Repo layout

```
align_book.py         The single alignment script
text/                 BSB text — one file per chapter, one verse per line
                      Filename: {BOOK}_{NNN}_BSB.txt   (e.g. JON_001_BSB.txt)
requirements.txt      torch, torchaudio, uroman
output/               (created on first run) word-timing JSON per chapter
```

The `text/` folder already contains the full BSB. You bring the audio.

---

## Setup

Python 3.9+ recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

First run downloads the MMS_FA model weights from torchaudio (~1 GB, cached).
The model runs on CPU; expect roughly real-time alignment on a recent laptop
(a 5-minute chapter takes a few minutes).

---

## Get the audio

The BSB is narrated by **Bob Hays** at openbible.com:

  https://openbible.com/audio/hays/

Download the per-chapter MP3s for the book you want into any directory
(e.g. `audio/hays/`). The script doesn't fetch them — that's on you.

The script matches audio files to chapters by globbing for the chapter
number in the filename. Common patterns work out of the box:

- `JON_001_*.mp3`
- `Jon_01.mp3`
- `BSB_32_Jonah_01_Hays.mp3`

If your filenames don't match, pass `--audio-glob` (see below).

---

## Usage

```bash
# Align every chapter of Jonah
python align_book.py --book JON --audio-dir audio/hays

# Just one chapter
python align_book.py --book JON --audio-dir audio/hays --chapter 1

# See which audio file would be picked, without aligning
python align_book.py --book JON --audio-dir audio/hays --dry-run

# Re-align even if output already exists
python align_book.py --book JON --audio-dir audio/hays --force
```

### Custom filename pattern

If auto-matching fails, pass an explicit pattern with placeholders:

```bash
python align_book.py --book JON --audio-dir audio/hays \
    --audio-glob "BSB_*_Jonah_{ch2}_*.mp3"
```

Available placeholders: `{book}` (uppercase), `{book_lc}` (lowercase),
`{ch}` (chapter as int), `{ch2}` (zero-padded to 2), `{ch3}` (zero-padded to 3).

### All flags

```
--book           3-letter BSB book code (e.g. JON, GEN, JHN)            [required]
--audio-dir      Directory containing per-chapter .mp3 files            [required]
--text-dir       BSB text directory                                     [default: text/]
--output-dir     Where to write timing JSON                             [default: output/]
--chapter        Process a single chapter only
--audio-glob     Custom glob pattern (see placeholders above)
--force          Overwrite existing output
--dry-run        Show what would be processed and exit
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

Run `ls text/ | cut -d_ -f1 | sort -u` for the full list.

---

## How it works

1. Load each `text/{BOOK}_{NNN}_BSB.txt`. One verse per line.
2. Strip punctuation, collapse whitespace. Concatenate the chapter into one string.
3. Romanize via `uroman` and tokenize with the MMS dictionary.
4. Run `torchaudio.pipelines.MMS_FA` forced alignment over the chapter audio.
   Long chapters are forwarded through the model in overlapping chunks
   (~5 min each at 16 kHz) so memory stays bounded.
5. The aligner returns one timed span per reference word — those words are
   1:1 with the cleaned text, so we slice them back into per-verse buckets
   by counting words per verse.
6. Write `output/{BOOK}/{BOOK}_{NNN}_words.json`.

There's no Whisper, no drift/header/collapse recovery, no quality file, and
no language config. Add them only if you need them.

---

## Troubleshooting

**`WARN no audio for JON 001`** — the glob didn't match. Run with `--dry-run`
and try `--audio-glob` with explicit placeholders.

**Low `avg_score` on a chapter** — usually means the audio has a long spoken
header (book/chapter title, music) that the alignment absorbs into the first
word. This script has no header detection — trim the leading intro from the
audio yourself, or accept that the first word's timing will be off.

**Alignment is slow** — MMS_FA runs on CPU here. A long chapter (Psalm 119,
Isaiah, etc.) can take many minutes. This is normal.

**Verse count looks off** — the BSB text files are one verse per line, and
empty lines are dropped. If your output has fewer verses than expected,
inspect the source `text/*.txt` for blank lines.

---

## License

Code: MIT — see `LICENSE`.

BSB text: public domain (Berean Standard Bible).
Hays audio: see openbible.com for terms.
