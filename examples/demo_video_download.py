from src.models.download_video import record_video

def main() -> None:
    youtube_url = "https://www.youtube.com/watch?v=tXhwFqhT8Ok"
    output_path = "./data/temp_videos/congress_example.mp4"
    duration_sec = 300  # 5 minutes

    record_video(
        youtube_url=youtube_url,
        output_path=output_path,
        duration_sec=duration_sec
    )

    print(f"Downloaded segment to {output_path}")


if __name__ == "__main__":
    main()
