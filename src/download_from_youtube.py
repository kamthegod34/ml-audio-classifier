from pathlib import Path
import yt_dlp

YOUTUBE_LINKS = {
    "car": [
        "https://www.youtube.com/watch?v=FIFBXz3MJmA",
        "https://www.youtube.com/watch?v=zeCJVMThL6k",
        "https://www.youtube.com/watch?v=bI08U8h5F7A"
    ],
    "drone": [
        "https://www.youtube.com/watch?v=DUTQkbuzxtk",
        "https://www.youtube.com/watch?v=ueHBOpN0Ikg",
        "https://www.youtube.com/watch?v=gA7-gb7l3Fc",
        "https://www.youtube.com/watch?v=qotZZxzCBaY",
        "https://www.youtube.com/watch?v=Ej8JOkQWqKo"
    ]
}

DEST = Path("data/raw")

def download_youtube_to_wav(url: str, dest: Path):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest.with_suffix(".%(ext)s")),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def main():
    for label, links in YOUTUBE_LINKS.items():
        label_direction = DEST / label
        for i, url in enumerate(links, start=1):
            out_file = label_direction / f"{label}_{i:04d}.wav"
            try:
                download_youtube_to_wav(url, out_file)
            except Exception as e:
                print(f"failed to download {url} due to problem: {e}")
                continue
    print("done!")

if __name__ == "__main__":
    main()
