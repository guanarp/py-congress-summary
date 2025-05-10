from __future__ import annotations

import time
from typing import Tuple
from llama_cpp import Llama
from .llm_utils import generate_response

def summarise(llm: Llama, text: str, *, max_tokens: int = 256) -> Tuple[str, float]:
    """Summarise *text* and return *(summary, elapsed_ms)*."""
    prompt = (
        "[INST] <<SYS>>Responde **EXCLUSIVAMENTE** en español. "
        "No utilices ningún otro idioma.<</SYS>>\n\n"
        f"Texto original:\n{text}\n\n"
        "Devuelve un resumen gramaticalmente correcto, coherente y de máximo 3 frases, "
        "mencionando al inicio quién es el autor.[/INST]"
    )

    start = time.perf_counter()
    summary = generate_response(llm, prompt, max_tokens)
    dur_ms = (time.perf_counter() - start) * 1000
    return summary, dur_ms
