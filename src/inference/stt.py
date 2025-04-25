from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
_MODEL_ID = "openai/whisper-large-v3-turbo"

_processor = AutoProcessor.from_pretrained(_MODEL_ID)
_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    _MODEL_ID,
    torch_dtype=_DTYPE,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
_model.to(_DEVICE)

_pipe = pipeline(
    "automatic-speech-recognition",
    model=_model,
    tokenizer=_processor.tokenizer,
    feature_extractor=_processor.feature_extractor,
    torch_dtype=_DTYPE,
    device=_DEVICE,
    return_timestamps=True,
)


def transcribe(paths: List[str] | str) -> List[str]:
    """Return raw transcripts for *paths* (preserves order)."""
    if isinstance(paths, (str, Path)):
        paths = [str(paths)]

    abs_paths: list[str] = []
    for p in paths:
        p = os.fspath(p)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        abs_paths.append(p)

    results = _pipe(abs_paths)
    
    if isinstance(results, dict):
        results = [results]
    return [r["text"] for r in results]

def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper v3‑turbo transcriber.",
    )
    parser.add_argument("audios", nargs="+", help="Audio file paths")
    args = parser.parse_args()

    textos = transcribe(args.audios)
    for path, txt in zip(args.audios, textos):
        name = Path(path).stem
        print(f"===== {name} =====")
        print(txt)
        print()


if __name__ == "__main__":  
    _main()