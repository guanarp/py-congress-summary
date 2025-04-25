from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

from llama_cpp import Llama

DEFAULT_N_CTX = 4096

_STOP_TOKENS = ["</s>", "[/INST]"]

def _guess_gpu_layers() -> int:
    """Heuristic: if the llama-cpp backend is CUDA/cuBLAS, load *all* layers.

    Returns
    -------
    int
        ``-1``  → llama.cpp loads the entire model on GPU.
        ``0``   → CPU build or unknown backend.
    """
    try:
        from llama_cpp import llama_cpp_get_backend  # type: ignore

        backend = llama_cpp_get_backend()
        print("Backend:", backend)
        return -1 if backend and "cuda" in backend.lower() else 0
    except Exception:
        # Older wheels don't expose the helper; fall back to 0.
        return 0



def load_model(path: str | Path, n_gpu_layers: int | None = None, *, n_ctx: int = DEFAULT_N_CTX,
               verbose: bool = False) -> Llama:
    """Load a GGUF checkpoint and return the *llama-cpp-python* object.

    Parameters
    ----------
    path
        Path to ``.gguf`` file.
    n_gpu_layers
        Number of layers to place on GPU. ``None`` (default) chooses ``-1`` if
        the wheel is CUDA‑enabled, else ``0``.
    n_ctx
        Context window passed to the model.
    verbose
        Whether to print the llama.cpp banner.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if n_gpu_layers is None:
        n_gpu_layers = _guess_gpu_layers()

    print("n_gpu_layers =",n_gpu_layers)

    return Llama(
        model_path=str(path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
    )


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
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.7,
        top_k=40,
        repeat_penalty=1.1,
        stop=_STOP_TOKENS,
    )
    dur_ms = (time.perf_counter() - start) * 1000
    summary = out["choices"][0]["text"].strip()
    return summary, dur_ms
