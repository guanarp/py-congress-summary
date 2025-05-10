from pathlib import Path
from typing import Any, Dict
from llama_cpp import Llama

DEFAULT_N_CTX = 4096
_STOP_TOKENS = ["</s>", "[/INST]"]

def _guess_gpu_layers() -> int:
    """Heuristic: if the llama-cpp backend is CUDA/cuBLAS, load *all* layers."""
    try:
        from llama_cpp import llama_cpp_get_backend  # type: ignore
        backend = llama_cpp_get_backend()
        print("Backend:", backend)
        return -1 if backend and "cuda" in backend.lower() else 0
    except Exception:
        return 0

def load_model(path: str | Path, n_gpu_layers: int | None = None, *, n_ctx: int = DEFAULT_N_CTX,
               verbose: bool = False) -> Llama:
    """Load a GGUF checkpoint and return the *llama-cpp-python* object."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if n_gpu_layers is None:
        n_gpu_layers = _guess_gpu_layers()

    print("n_gpu_layers =", n_gpu_layers)

    return Llama(
        model_path=str(path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=verbose,
    )

def generate_response(llm: Llama, prompt: str, max_tokens: int = 256) -> str:
    """Generate a response from the LLM with consistent parameters."""
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.7,
        top_k=40,
        repeat_penalty=1.1,
        stop=_STOP_TOKENS,
    )
    return response["choices"][0]["text"].strip() 