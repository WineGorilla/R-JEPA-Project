from pathlib import Path
import os
import urllib.request
import zipfile

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA/data/clevrer")

SPLITS = {
    "validation": {
        "url": "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
        "zip_name": "video_validation.zip",
    },
    "test": {
        "url": "http://data.csail.mit.edu/clevrer/videos/test/video_test.zip",
        "zip_name": "video_test.zip",
    },
}


def download_file(url, output_path):
    if output_path.exists():
        print(f"ZIP already exists: {output_path}")
        return

    print(f"Downloading:")
    print(url)

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size

        if total_size > 0:
            percent = min(
                downloaded / total_size * 100,
                100.0,
            )

            print(
                f"\r{percent:6.2f}% "
                f"({downloaded / 1024**3:.2f} / "
                f"{total_size / 1024**3:.2f} GiB)",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(
        url,
        output_path,
        reporthook=progress,
    )

    print()


def extract_zip(zip_path, output_dir):
    marker = output_dir / ".extract_complete"

    if marker.exists():
        print(f"Already extracted: {output_dir}")
        return

    print(f"Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    marker.touch()

    print("Extraction complete.")


def flatten_mp4_files(split_dir):
    mp4_files = list(
        split_dir.rglob("*.mp4")
    )

    print(
        f"Found {len(mp4_files)} mp4 files."
    )

    for source in mp4_files:
        if source.parent == split_dir:
            continue

        target = split_dir / source.name

        if target.exists():
            continue

        os.replace(
            source,
            target,
        )

    for path in sorted(
        split_dir.iterdir()
    ):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def count_videos(split_dir):
    return len(
        list(
            split_dir.glob("*.mp4")
        )
    )


def main():
    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("CLEVRER validation + test download")
    print("=" * 80)
    print("Destination:")
    print(ROOT)
    print()

    for split, config in SPLITS.items():
        print("=" * 80)
        print(split.upper())
        print("=" * 80)

        split_dir = ROOT / split

        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        zip_path = (
            ROOT / config["zip_name"]
        )

        download_file(
            config["url"],
            zip_path,
        )

        extract_zip(
            zip_path,
            split_dir,
        )

        flatten_mp4_files(
            split_dir
        )

        count = count_videos(
            split_dir
        )

        print(
            f"{split} videos: {count}"
        )

        print()

    print("=" * 80)
    print("DOWNLOAD COMPLETE")
    print("=" * 80)

    print(
        "Validation:",
        ROOT / "validation",
    )

    print(
        "Test:",
        ROOT / "test",
    )


if __name__ == "__main__":
    main()