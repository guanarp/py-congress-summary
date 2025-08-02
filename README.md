# py-congress-summary

**Work in Progress**: This project is still under development. Many components are functional and modular, but active improvements are ongoing.

---

## Overview

**py-congress-summary** is a modular and extensible pipeline for transcribing, analyzing, and summarizing long-form institutional sessions like parliamentary debates, hearings, or public discussions. The system handles the full flow: video ingestion, transcription, semantic chunking, speaker-aware detection, and summarization using local LLMs.

---

## Features

- **Video Capture**: Extracts segments from livestreams or YouTube videos.
- **Speech-to-Text**: High-accuracy transcription using Whisper (large v3 turbo).
- **Semantic Chunking**: Breaks transcripts into coherent units.
- **Speaker Detection**: LLM-based speaker change identification.
- **Summarization**: Local LLM summarizes each speaker-tagged chunk.
- **Modular Output**: JSON, plain text, or integration-ready formats.

---

## Tech Stack

| Layer         | Tool                                  |
| ------------- | ------------------------------------- |
| STT           | `openai/whisper-large-v3-turbo`       |
| LLM           | GGUF model via llama.cpp or HF        |
| Video         | `yt-dlp` + `ffmpeg`                   |
| Chunking      | Custom semantic and speaker splitters |
| Summarization | Prompt-based LLM summarizer           |

---

## Pipeline Overview

### 1. Video Capture

```bash
python examples/demo_video_download.py --url <video_url> --duration 120
```

Downloads a 2-minute segment using `yt-dlp` + `ffmpeg`.

### 2. Transcription

```bash
python examples/demo_transcribe.py --input segment_2min.mp4 --language es
```

Uses Whisper to generate a timestamped Spanish transcript.

### 3. Semantic Chunking

```python
from semantic_splitter import SemanticSplitter
chunks = SemanticSplitter().split_text(transcription)
```

Creates coherent discussion units for further analysis.

### 4. Speaker-Aware Splitting

```python
from speaker_splitter import SpeakerAwareSplitter
chunks_with_speakers = SpeakerAwareSplitter(llm=model).split_texts(chunks)
```

LLM detects and splits chunks based on speaker shifts.

### 5. Summarization

```python
from summarizer import summarise
summary, runtime = summarise(llm, chunk)
```

Each chunk is summarized with attribution to the correct speaker.

### 6. Output Example

```json
[
  {
    "speaker": "Senador López",
    "summary": "Propuso modificar el artículo 13 para incluir representación indígena en decisiones regionales.",
    "timestamp": "00:02:15"
  }
]
```

---

## Setup

```bash
git clone https://github.com/yourname/py-congress-summary.git
cd py-congress-summary
python -m venv congress_venv
source congress_venv/bin/activate  # Windows: .\congress_venv\Scripts\activate
pip install -r requirements.txt
```

Install system dependencies:

```bash
sudo apt install ffmpeg
pip install yt-dlp
```

---

## To-Do & Roadmap

- Make better the LLM-based speaker change detection
  - Explore using diarization instead
- Enable summarization of full-length sessions with multi-speaker attribution
- Deploy backend to the cloud with trigger-based automation for session processing
- Build a public-facing web interface to browse meeting summaries and quotes by each senador/diputado
- Add integration with X/Twitter to automatically post highlights from sessions
- Ensure all steps run with minimal human interaction (fully automated flow)
- Scale the idea to other type of meetings and maybe expand to multilingual summarization capabilities

---

## Author

Built by **Jose Carlos Rios Parquet** to enable automated, speaker-aware analysis of institutional discourse.

> Feedback and collaboration welcome!

