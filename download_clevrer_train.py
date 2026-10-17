from pathlib import Path
import argparse
import shutil
import zipfile

import requests

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
DATA_ROOT = ROOT / "data/clevrer"
TRAIN_DIR = DATA_ROOT / "train"
ZIP_PATH = DATA_ROOT / "video_train.zip"

URL = "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip"
CHUNK_SIZE = 8 * 1024 * 1024


def download(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)

    existing = out.stat().st_size if out.exists() else 0
    headers = {}
    mode = "wb"

    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"Resuming from {existing / (1024**3):.2f} GB")

    r = requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=(30, 300),
        allow_redirects=True,
    )

    if existing > 0 and r.status_code != 206:
        print("Server does not support resume. Restarting download.")
        r.close()
        existing = 0
        mode = "wb"

        r = requests.get(
            url,
            stream=True,
            timeout=(30, 300),
            allow_redirects=True,
        )

    r.raise_for_status()

    total = r.headers.get("Content-Length")
    total = int(total) + existing if total else None

    downloaded = existing

    with out.open(mode) as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue

            f.write(chunk)
            downloaded += len(chunk)

            if total:
                pct = downloaded / total * 100
                print(
                    f"\rDownloaded "
                    f"{downloaded / (1024**3):.2f} / "
                    f"{total / (1024**3):.2f} GB "
                    f"({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rDownloaded "
                    f"{downloaded / (1024**3):.2f} GB",
                    end="",
                    flush=True,
                )

    print()
    print("Download complete:", out)


def extract_train(zip_path, train_dir):
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Invalid zip file: {zip_path}")

    temp_dir = DATA_ROOT / "_train_extract_tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True)

    print("Extracting CLEVRER training videos...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    mp4s = sorted(temp_dir.rglob("*.mp4"))

    if not mp4s:
        raise RuntimeError("No mp4 files found after extraction.")

    print(f"Found {len(mp4s)} videos.")

    train_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(mp4s, 1):
        dst = train_dir / src.name

        if not dst.exists():
            shutil.move(str(src), str(dst))

        if i % 500 == 0 or i == len(mp4s):
            print(f"Moved {i}/{len(mp4s)} videos")

    shutil.rmtree(temp_dir)

    count = len(list(train_dir.glob("*.mp4")))

    print()
    print("Final train video count:", count)

    first = train_dir / "video_00000.mp4"

    if first.exists():
        print("Sanity check OK:", first)
    else:
        print("Warning: video_00000.mp4 not found.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep zip after extraction",
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download, do not extract",
    )

    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    existing_count = (
        len(list(TRAIN_DIR.glob("*.mp4")))
        if TRAIN_DIR.exists()
        else 0
    )

    if existing_count >= 10000:
        print(
            f"CLEVRER training videos already exist: "
            f"{existing_count}"
        )
        print(TRAIN_DIR)
        return

    print("=" * 80)
    print("CLEVRER TRAIN VIDEO DOWNLOADER")
    print("=" * 80)
    print("URL:", URL)
    print("Zip path:", ZIP_PATH)
    print("Train directory:", TRAIN_DIR)

    download(URL, ZIP_PATH)

    if args.download_only:
        return

    extract_train(
        ZIP_PATH,
        TRAIN_DIR,
    )

    if not args.keep_zip:
        ZIP_PATH.unlink(missing_ok=True)
        print("Removed zip file.")

    print()
    print("Done.")
    print("Training videos:")
    print(TRAIN_DIR)


if __name__ == "__main__":
    main()