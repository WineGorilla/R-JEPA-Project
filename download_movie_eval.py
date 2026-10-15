import os
import time
from pathlib import Path

import gcsfs

SRC = "kubric-public/tfds/movi_e/256x256/1.0.0"
DST = Path(
    "/data/mi2-interns/ruiyu/R-JEPA/"
    "data/tfds/movi_e/256x256/1.0.0"
)

DST.mkdir(parents=True, exist_ok=True)

fs = gcsfs.GCSFileSystem(token="anon")

print("=" * 80)
print("MOVi-E validation + test download")
print("=" * 80)
print("Source:")
print(SRC)
print()
print("Destination:")
print(DST)
print()

print("Scanning remote files...")

entries = fs.find(
    SRC,
    detail=True,
)

files = []

for remote_path, info in entries.items():
    remote_path = remote_path.rstrip("/")

    if remote_path == SRC.rstrip("/"):
        continue

    if info.get("type") == "directory":
        continue

    rel_path = os.path.relpath(
        remote_path,
        SRC,
    )

    if rel_path == "." or rel_path.startswith(".."):
        continue

    name = os.path.basename(remote_path)

    keep = False

    # --------------------------------------------------------
    # TFDS metadata files
    # --------------------------------------------------------

    if name in {
        "dataset_info.json",
        "features.json",
        "instances-category.labels.txt",
    }:
        keep = True

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    if "validation" in name:
        keep = True

    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------

    if "test" in name:
        keep = True

    if keep:
        files.append(
            (
                remote_path,
                rel_path,
                int(info.get("size", 0)),
            )
        )

files.sort(
    key=lambda x: x[1]
)

total_size = sum(
    size
    for _, _, size in files
)

print(
    f"Found {len(files)} files"
)

print(
    f"Total size: "
    f"{total_size / 1024**3:.2f} GiB"
)

print()


downloaded_before = 0

for remote_path, rel_path, remote_size in files:
    local_path = DST / rel_path
    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if local_path.exists():
        local_size = local_path.stat().st_size

        if local_size == remote_size:
            print(
                f"SKIP {rel_path} "
                f"({remote_size / 1024**3:.3f} GiB)"
            )

            downloaded_before += remote_size
            continue

    part_path = Path(
        str(local_path) + ".part"
    )

    if part_path.exists():
        part_path.unlink()

    index = files.index(
        (
            remote_path,
            rel_path,
            remote_size,
        )
    ) + 1

    print(
        f"[{index}/{len(files)}] "
        f"Downloading {rel_path} "
        f"({remote_size / 1024**3:.3f} GiB)"
    )

    start = time.time()

    with fs.open(
        remote_path,
        "rb",
    ) as src_file:

        with open(
            part_path,
            "wb",
        ) as dst_file:

            while True:
                chunk = src_file.read(
                    8 * 1024 * 1024
                )

                if not chunk:
                    break

                dst_file.write(
                    chunk
                )

    local_size = part_path.stat().st_size

    if local_size != remote_size:
        raise RuntimeError(
            f"Size mismatch for {rel_path}: "
            f"remote={remote_size}, "
            f"local={local_size}"
        )

    os.replace(
        part_path,
        local_path,
    )

    elapsed = time.time() - start

    speed = (
        remote_size
        / 1024**2
        / max(elapsed, 1e-6)
    )

    downloaded_before += remote_size

    progress = (
        downloaded_before
        / total_size
        * 100
    )

    print(
        f"    done | "
        f"{elapsed:.1f}s | "
        f"{speed:.1f} MiB/s | "
        f"total progress "
        f"{progress:.2f}%"
    )


print()
print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)

print(
    "Downloaded size:",
    f"{total_size / 1024**3:.2f} GiB"
)

print(
    "Local dataset path:"
)

print(DST)