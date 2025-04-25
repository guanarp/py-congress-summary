from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download(repo_id: str, allow_patterns: str, out_dir: str) -> Path:
    """Download model files that match *allow_patterns* into *out_dir*.

    Returns
    -------
    Path
        Local directory where the snapshot was stored.
    """
    dest = Path(out_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow_patterns,
        local_dir=dest,
        token=os.getenv("HF_TOKEN"),
    )
    return Path(snapshot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GGUF model from HF Hub")
    parser.add_argument("--repo", required=True, help="HF repository id")
    parser.add_argument("--pattern", default="*gguf", help="glob pattern to allow")
    parser.add_argument("--out", default=f"./data/models/", help="output directory")
    args = parser.parse_args()

    path = download(args.repo, args.pattern, args.out)
    print(f"Model downloaded to: {path}")