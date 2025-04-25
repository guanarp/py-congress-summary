import argparse

from src.inference.stt import transcribe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR demo (Whisper large‑v3‑turbo)")
    parser.add_argument("audio", help="Audio file path")
    args = parser.parse_args()

    print(transcribe(args.audio)[0])