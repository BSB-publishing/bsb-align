PY ?= python3
AUDIO_DIR ?= audio
BOOK ?=
CHAPTER ?=
AUDIO_GLOB ?=

CHAPTER_FLAG = $(if $(CHAPTER),--chapter $(CHAPTER))
GLOB_FLAG    = $(if $(AUDIO_GLOB),--audio-glob "$(AUDIO_GLOB)")

.PHONY: help install install-whisper whisper mms align book clean

help:
	@echo "Targets:"
	@echo "  install          Install MMS deps (torch, torchaudio, uroman)"
	@echo "  install-whisper  Install mlx-whisper (Apple Silicon, optional)"
	@echo "  whisper          Step 1a: Whisper transcription   (BOOK=JON [CHAPTER=1] [AUDIO_DIR=...])"
	@echo "  mms              Step 1b: MMS forced alignment    (BOOK=JON [CHAPTER=1] [AUDIO_DIR=...])"
	@echo "  align            Step 2:  fuse Whisper + MMS      (BOOK=JON [CHAPTER=1] [AUDIO_DIR=...])"
	@echo "  book             Run all three for one book       (BOOK=JON [CHAPTER=1] [AUDIO_DIR=...])"
	@echo "  clean            Remove output/"
	@echo ""
	@echo "Examples:"
	@echo "  make book BOOK=JON"
	@echo "  make book BOOK=JHN AUDIO_DIR=audio/hays"
	@echo "  make mms BOOK=GEN CHAPTER=1"

_check_book:
	@if [ -z "$(BOOK)" ]; then echo "Error: BOOK is required (e.g. make $(MAKECMDGOALS) BOOK=JON)"; exit 1; fi

install:
	pip install -r requirements.txt

install-whisper:
	pip install -r requirements-whisper.txt

whisper: _check_book
	$(PY) whisper_transcribe.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG)

mms: _check_book
	$(PY) mms_align_words.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG)

align: _check_book
	$(PY) align_words.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG)

book: _check_book
	$(PY) whisper_transcribe.py --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG)
	$(PY) mms_align_words.py    --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG)
	$(PY) align_words.py        --book $(BOOK) --audio-dir $(AUDIO_DIR) $(CHAPTER_FLAG) $(GLOB_FLAG)

clean:
	rm -rf output
