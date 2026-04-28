PY ?= python3
AUDIO_DIR ?= audio
BOOK ?=
CHAPTER ?=
AUDIO_GLOB ?=
FORCE ?=

CHAPTER_FLAG = $(if $(CHAPTER),--chapter $(CHAPTER))
GLOB_FLAG    = $(if $(AUDIO_GLOB),--audio-glob "$(AUDIO_GLOB)")
FORCE_FLAG   = $(if $(FORCE),--force)

.PHONY: help install install-whisper whisper mms align book upgrade clean

help:
	@echo "Targets:"
	@echo "  install          Install MMS deps (torch, torchaudio, numpy, uroman, torchcodec)"
	@echo "  install-whisper  Install mlx-whisper (Apple Silicon, optional)"
	@echo "  whisper          Step 1a: Whisper transcription"
	@echo "  mms              Step 1b: MMS forced alignment"
	@echo "  align            Step 2:  fuse Whisper + MMS"
	@echo "  book             Run all three steps for one book"
	@echo "  upgrade          Re-run all three steps with --force (overwrites existing _words.json)"
	@echo "  clean            Remove output/"
	@echo ""
	@echo "Variables:"
	@echo "  BOOK=JON         (required) 3-letter BSB book code"
	@echo "  CHAPTER=1        Limit to a single chapter"
	@echo "  AUDIO_DIR=audio  Per-chapter mp3 directory (default: audio/)"
	@echo "  AUDIO_GLOB=...   Custom glob pattern"
	@echo "  FORCE=1          Pass --force to all steps (overwrite existing output)"
	@echo ""
	@echo "Examples:"
	@echo "  make book BOOK=JON"
	@echo "  make book BOOK=JHN AUDIO_DIR=audio/hays"
	@echo "  make mms BOOK=GEN CHAPTER=1"
	@echo "  make upgrade BOOK=ROM   # re-fuse a book that has older _words.json"

_check_book:
	@if [ -z "$(BOOK)" ]; then echo "Error: BOOK is required (e.g. make $(MAKECMDGOALS) BOOK=JON)"; exit 1; fi

install:
	pip install -r requirements.txt

install-whisper:
	pip install -r requirements-whisper.txt

whisper: _check_book
	$(PY) whisper_transcribe.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG) $(FORCE_FLAG)

mms: _check_book
	$(PY) mms_align_words.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG) $(FORCE_FLAG)

align: _check_book
	$(PY) align_words.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG) $(FORCE_FLAG)

book: _check_book
	$(PY) whisper_transcribe.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG) $(FORCE_FLAG)
	$(PY) mms_align_words.py    --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG) $(FORCE_FLAG)
	$(PY) align_words.py        --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG) $(FORCE_FLAG)

upgrade: _check_book
	$(MAKE) book BOOK=$(BOOK) CHAPTER=$(CHAPTER) AUDIO_DIR=$(AUDIO_DIR) AUDIO_GLOB=$(AUDIO_GLOB) FORCE=1

clean:
	rm -rf output
