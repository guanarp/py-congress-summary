from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.speaker_splitter_llm import SpeakerDetectorLLM, SpeakerAwareSplitter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speaker detection and splitting demo")
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
    return parser.parse_args()


def main() -> None: 
    args = _parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run examples/demo_download.py first."
        )

    # Initialize the speaker detector and splitter
    speaker_detector = SpeakerDetectorLLM(model_path, n_gpu_layers=args.gpu_layers)
    splitter = SpeakerAwareSplitter(speaker_detector)

    # Define the chunks
    chunks = [
        "Gracias. Bienvenidos sean todos a este lanzamiento del libro \"Paraguayas en el poder\". ",

        "Seguidamente quiero convocar a la doctora Estela González de Rojas, prologuista, a quien recibimos con un fuerte aplauso.",

        "Muy buenos días a todos. Es un honor estar aquí y poder compartir unas palabras sobre esta obra tan importante.",

        "Las mujeres en Paraguay han demostrado una capacidad increíble para liderar y transformar.",

        "Ahora, vamos a escuchar a Nilsa Maíz de Sotomayor, autora del libro, quien nos compartirá su visión.",

        "Gracias, muchas gracias. Para mí, este libro representa un sueño hecho realidad.",

        "Es una forma de inmortalizar las historias de mujeres que cambiaron nuestro país.",

        "Finalmente, quiero invitar a la diputada Rocío Abed de Zacarías, coautora del libro.",

        "Buenos días, querida audiencia. Este libro es más que una recopilación de historias,",

        "Es una prueba del coraje y la determinación de las mujeres paraguayas.",

        "Gracias a todas las mujeres que participaron y a todos ustedes por acompañarnos hoy.",

        "Señor Presidente, muchas gracias. Quiero comenzar recordando a todos los presentes que hoy es un día histórico para nuestro país. En segundo lugar, quiero felicitar a todos los miembros del Congreso por su dedicación y arduo trabajo durante estos últimos meses. Ahora bien, es importante destacar que aún queda mucho por hacer para garantizar el bienestar de nuestros ciudadanos.",
        
        "No basta con aprobar leyes; debemos asegurarnos de que sean implementadas de manera efectiva en cada rincón del país. Además, propongo que se creen comisiones especiales para abordar estos desafíos de forma integral, involucrando tanto al sector público como al privado. Permítanme también subrayar el rol de las nuevas generaciones. Los jóvenes necesitan sentirse parte de este proceso democrático.",
        
        "Propongo la creación de foros juveniles en cada departamento para que puedan expresar sus ideas y preocupaciones. En el ámbito económico, debemos redoblar nuestros esfuerzos para fomentar la inversión extranjera, sin descuidar nuestras industrias locales. Es vital diseñar políticas públicas que sean inclusivas y que promuevan el crecimiento sostenible. Por otro lado, no podemos ignorar el impacto del cambio climático.",
        
        "Propongo establecer una comisión bicameral dedicada exclusivamente al desarrollo de estrategias de mitigación y adaptación. Cada región debe ser escuchada para diseñar soluciones acordes a sus realidades particulares. Finalmente, quiero hacer un llamado a la unidad. Más allá de nuestras diferencias políticas, compartimos un mismo objetivo: construir un país más justo, equitativo y próspero.",
        
        "Agradezco profundamente el apoyo y la colaboración de cada uno de ustedes en este desafío histórico."
    ]
    
    # Process the chunks and get speaker-labeled segments
    results = splitter.split_chunks(chunks, starting_speaker="Maestro de Ceremonias")
    
    # Print the results
    print("Speaker Analysis Results:")
    print("-" * 50)
    for speaker, text in results:
        print(f"Next peaker: {speaker}")
        print(f"Text: {text}")
        print("-" * 50)


if __name__ == "__main__":
    main() 