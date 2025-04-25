from src.models.download_mistral import download

if __name__ == "__main__":
    path = download(
        repo_id="ecastera/eva-mistral-7b-spanish-GGUF",
        allow_patterns="*int4*.gguf",
        out_dir="./data/models/eva_gguf",
    )
    print(f"Model stored at: {path}")