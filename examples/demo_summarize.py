from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.summarizer import load_model, summarise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single‑paragraph summariser demo")
    parser.add_argument(
        "--model",
        default="./data/models/eva_gguf/Turdus-trained-20-int4.gguf",
        help="Path to the GGUF checkpoint",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=None,
        help=(
            "How many transformer layers to off‑load to the GPU. "
            "Use ‑1 for *all* layers, 0 to force CPU. If omitted a sane "
            "default is chosen based on the backend (CUDA vs. CPU build)."
        ),
    )
    parser.add_argument(
        "--text",
        default=(
            "El Ministerio de Economía de Paraguay anunció hoy un paquete de medidas para mitigar los "
            "efectos de la sequía sobre los pequeños productores. Entre las acciones se incluye la prórroga "
            "de créditos, la entrega de semillas y fertilizantes subsidiados, y la creación de un fondo de "
            "emergencia de 100 millones de dólares."
        ),
        help="Text to summarise (if omitted a built‑in sample is used)",
    )
    return parser.parse_args()


def main() -> None: 
    args = _parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run examples/demo_download.py first."
        )

    llm = load_model(model_path)
    print("backend =", llm.metadata.get("backend", "cpu"))
    summary, ms = summarise(llm, args.text)
    print(summary)
    print(f"Inference time: {ms:.0f} ms")


if __name__ == "__main__":
    main()