"""Shared text processing for the BSB alignment pipeline.

Slim version — extracted from the multi-language pipeline. The dataclass
shape is preserved so the algorithmic code in align_words.py and
mms_align_words.py works unchanged, but this repo only ever uses the
default config (English BSB, no markers, no pronunciation maps).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LanguageConfig:
    """Language-specific text processing configuration.

    For BSB the defaults are correct — `re.sub(r"[^\\w\\s]", "", text)` in
    clean_for_alignment already strips smart quotes, em dashes, and other
    punctuation, so no char replacements or pronunciation maps are needed.
    """
    iso: str = "eng"
    pronunciation_map: Dict[str, str] = field(default_factory=dict)
    strip_marker_rules: List[Dict[str, str]] = field(default_factory=list)
    char_replacements: Dict[str, str] = field(default_factory=dict)
    strip_unicode_categories: List[str] = field(default_factory=lambda: ["Mn"])
    mms_fallback_threshold: float = 0.3
    aramaic_passages: List[str] = field(default_factory=list)


_DEFAULT_CONFIG = LanguageConfig()


def load_language_config(iso: str = "eng") -> LanguageConfig:
    """Return the default English BSB config. The iso arg is accepted for
    compatibility with the upstream pipeline but ignored."""
    return _DEFAULT_CONFIG


def is_aramaic_chapter(book: str, chapter: int, config: LanguageConfig) -> bool:
    """Always False for English BSB."""
    return False


def strip_markers(text: str, config: LanguageConfig) -> str:
    """No-op for English BSB (no marker rules configured)."""
    for rule in config.strip_marker_rules:
        text = re.sub(rule["pattern"], rule.get("replacement", ""), text)
    return text.strip()


def clean_for_alignment(text: str, config: LanguageConfig) -> str:
    """Clean text for forced alignment and word counting.

    Used identically by mms_align_words.py (to feed MMS-FA) and align_words.py
    (to count words per verse). Steps: strip diacritics, apply char
    replacements, apply pronunciation map, strip punctuation, collapse
    whitespace. Does NOT lowercase.
    """
    categories = set(config.strip_unicode_categories)
    text = "".join(c for c in text if unicodedata.category(c) not in categories)
    for old, new in config.char_replacements.items():
        text = text.replace(old, new)
    for original, replacement in config.pronunciation_map.items():
        text = text.replace(original, replacement)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str, config: LanguageConfig) -> str:
    """Lowercased version of clean_for_alignment. For fuzzy matching."""
    return clean_for_alignment(text.lower(), config)
