"""Audio file matching for BSB chapters.

Per-chapter audio files (e.g. from openbible.com/audio/hays/) don't follow
a single naming convention. This helper tries several common patterns and
consults a small alias map for books where Hays uses an unusual stem.

Used by align_book.py, export_verses.py, mms_align_words.py, and
whisper_transcribe.py so all four scripts agree on what audio matches
which chapter.
"""

from pathlib import Path
from typing import Optional


# Books where Hays-style filenames don't match the standard 3-letter code.
# Keys are uppercase BSB codes; values are the alternative stem to search for.
BOOK_ALIAS = {
    "TIT": "Tts",
}


def find_audio(audio_dir: Path, book: str, chapter: int,
               glob_template: Optional[str] = None,
               recursive: bool = False) -> Optional[Path]:
    """Find an audio file for a book/chapter inside audio_dir.

    Tries glob_template first if given (with `{book}`, `{book_lc}`,
    `{book_title}`, `{ch}`, `{ch2}`, `{ch3}` placeholders), then falls back
    to standard patterns over the book code in upper/lower/title case, plus
    an alias if BOOK_ALIAS has one.

    `recursive=True` searches subdirectories too (rglob); default is non-recursive.
    """
    book = book.upper()
    book_lc = book.lower()
    book_title = book.title()
    book_alias = BOOK_ALIAS.get(book)
    ch2 = f"{chapter:02d}"
    ch3 = f"{chapter:03d}"

    fmt_args = {
        "book": book,
        "book_lc": book_lc,
        "book_title": book_title,
        "ch": chapter,
        "ch2": ch2,
        "ch3": ch3,
    }

    patterns = []
    if glob_template:
        patterns.append(glob_template.format(**fmt_args))

    for stem in (book, book_lc, book_title):
        patterns.append(f"*{stem}*{ch3}*.mp3")
        patterns.append(f"*{stem}*{ch2}*.mp3")

    if book_alias:
        patterns.append(f"*{book_alias}*{ch3}*.mp3")
        patterns.append(f"*{book_alias}*{ch2}*.mp3")

    seen = set()
    glob_fn = audio_dir.rglob if recursive else audio_dir.glob
    for pat in patterns:
        if pat in seen:
            continue
        seen.add(pat)
        candidates = sorted(glob_fn(pat))
        if candidates:
            return candidates[0]
    return None
