import argparse

from src.inference.stt import transcribe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR demo (Whisper large‑v3‑turbo)")
    parser.add_argument("audio", help="Audio file path")
    args = parser.parse_args()

    transcription, inference_time =  transcribe(args.audio)

    print(transcription[0])
    print(f"Inference time: {inference_time:0f} ms")