from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List
import time

import torch

from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    pipeline
)

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
_MODEL_ID = "openai/whisper-large-v3-turbo"

_feature_extractor = WhisperFeatureExtractor.from_pretrained(_MODEL_ID)
_tokenizer = WhisperTokenizer.from_pretrained(_MODEL_ID, language="spanish", task="transcribe")
_model = WhisperForConditionalGeneration.from_pretrained(_MODEL_ID)
_forced_decoder_ids = _tokenizer.get_decoder_prompt_ids(language="spanish", task="transcribe")

_model.to(_DEVICE)

_pipe = pipeline(
    "automatic-speech-recognition",
    model=_model,
    tokenizer=_tokenizer,
    feature_extractor=_feature_extractor,
    torch_dtype=_DTYPE,
    device=_DEVICE,
    chunk_length_s=30,
    return_timestamps=True,
    #stride_length_s=(4, 2) # No estoy seguro de si este necesito para tener el algoritmo de HF
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
    
    # Start the timer
    start_time = time.time()

    results = _pipe(abs_paths, generate_kwargs={"forced_decoder_ids": _forced_decoder_ids})

    # Calculate inference time
    inference_time = time.time() - start_time
    
    if isinstance(results, dict):
        results = [results]

    transcriptions = [r["text"] for r in results]

    return transcriptions, inference_time

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