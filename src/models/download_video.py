import subprocess
import argparse


def record_video(
    youtube_url: str,
    output_path: str,
    duration_sec: int | None = None,
) -> None:
    """
    Records a segment from a live YouTube URL into output_path.

    If duration_sec is provided, records for that many seconds.
    Otherwise, records until manually stopped or stream ends.

    Parameters
    ----------
    youtube_url : str
        The YouTube live URL or video URL to record from.
    output_path : str
        The file path where the recording will be saved.
    duration_sec : int or None
        Duration in seconds to record. If None, records indefinitely.
    """
    cmd = [
        "yt-dlp",
        youtube_url,
        "--external-downloader", "ffmpeg",
        "-x",  # Extract audio only
        "--audio-format", "mp3",  # Specify the audio format as mp3
    ]

    ffmpeg_args = ["-nostdin"]
    if duration_sec is not None:
        ffmpeg_args += ["-ss", "0", "-t", str(duration_sec)]

    cmd += ["--external-downloader-args", " ".join(ffmpeg_args), "-o", str(output_path)]

    subprocess.run(cmd, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a YouTube video segment using yt-dlp and ffmpeg.")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="The YouTube video URL to download."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The output file path for the recording."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Optional duration in seconds to record. If omitted, records full stream."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record_video(args.url, args.output, args.duration)


if __name__ == "__main__":
    main()
