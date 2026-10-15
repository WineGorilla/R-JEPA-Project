import os
import time
import gcsfs


# ============================================================
# Config
# ============================================================

SRC = (
    "kubric-public/"
    "tfds/"
    "movi_c/"
    "256x256/"
    "1.0.0"
)

DST = (
    "/data/mi2-interns/ruiyu/"
    "R-JEPA/data/tfds/"
    "movi_c/256x256/1.0.0"
)


# ============================================================
# Setup
# ============================================================

os.makedirs(
    DST,
    exist_ok=True,
)


fs = gcsfs.GCSFileSystem(
    token="anon"
)


print("=" * 80)

print(
    "MOVi-C download"
)

print("=" * 80)


print(
    "Source:"
)

print(
    SRC
)


print()


print(
    "Destination:"
)

print(
    DST
)


print()


# ============================================================
# Find remote objects
# ============================================================

print(
    "Scanning remote files..."
)


all_entries = fs.find(
    SRC,
    detail=True,
)


# ============================================================
# IMPORTANT:
#
# fs.find() may include the root directory itself.
#
# We only keep actual GCS files / objects.
# ============================================================

files = []


for remote_path, info in all_entries.items():

    # Ignore source root
    if (
        remote_path.rstrip("/")
        == SRC.rstrip("/")
    ):
        continue


    # Ignore directories
    if (
        info.get("type")
        == "directory"
    ):
        continue


    # Keep only real files
    files.append(
        (
            remote_path,
            info,
        )
    )


# Sort for reproducible order
files.sort(
    key=lambda x: x[0]
)


print(
    f"Found {len(files)} actual files"
)


# ============================================================
# Calculate total size
# ============================================================

total_bytes = sum(

    info.get(
        "size",
        0,
    )

    for _, info in files
)


print(
    f"Total remote size: "
    f"{total_bytes / 1024**3:.2f} GiB"
)


print()


# ============================================================
# Download
# ============================================================

downloaded_bytes = 0

start_all = time.time()


for index, (
    remote_path,
    info,
) in enumerate(
    files,
    start=1,
):

    # --------------------------------------------------------
    # Relative path
    # --------------------------------------------------------

    rel_path = os.path.relpath(
        remote_path,
        SRC,
    )


    # Extra safety:
    # Never accept "." or paths outside SRC.
    if (
        rel_path == "."
        or rel_path.startswith("..")
    ):

        print(
            f"[{index}/{len(files)}] "
            f"IGNORE invalid path: "
            f"{rel_path}"
        )

        continue


    local_path = os.path.join(
        DST,
        rel_path,
    )


    parent_dir = os.path.dirname(
        local_path
    )


    os.makedirs(
        parent_dir,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Remote size
    # --------------------------------------------------------

    remote_size = info.get(
        "size",
        0,
    )


    # ========================================================
    # Resume:
    #
    # Existing file with identical size -> skip.
    # ========================================================

    if os.path.isfile(
        local_path
    ):

        local_size = os.path.getsize(
            local_path
        )


        if (
            local_size
            == remote_size
        ):

            downloaded_bytes += (
                remote_size
            )


            print(
                f"[{index}/{len(files)}] "
                f"SKIP "
                f"{rel_path}"
            )

            continue


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        f"[{index}/{len(files)}] "
        f"Downloading "
        f"{rel_path}"
        f" "
        f"({remote_size / 1024**3:.3f} GiB)"
    )


    # --------------------------------------------------------
    # Temporary file
    # --------------------------------------------------------

    temp_path = (
        local_path
        + ".part"
    )


    # If previous attempt left partial file,
    # remove partial object and retry this file.
    if os.path.exists(
        temp_path
    ):

        if os.path.isdir(
            temp_path
        ):

            raise RuntimeError(
                f"Unexpected directory at "
                f"{temp_path}"
            )

        os.remove(
            temp_path
        )


    # --------------------------------------------------------
    # Download one object
    # --------------------------------------------------------

    file_start = time.time()


    fs.get(
        remote_path,
        temp_path,
    )


    # --------------------------------------------------------
    # Verify size
    # --------------------------------------------------------

    local_temp_size = os.path.getsize(
        temp_path
    )


    if (
        local_temp_size
        != remote_size
    ):

        raise RuntimeError(
            "\nFile size mismatch!\n"
            f"Remote: {remote_size}\n"
            f"Local : {local_temp_size}\n"
            f"File  : {remote_path}"
        )


    # --------------------------------------------------------
    # Atomic rename
    # --------------------------------------------------------

    os.replace(
        temp_path,
        local_path,
    )


    elapsed = (
        time.time()
        - file_start
    )


    downloaded_bytes += (
        remote_size
    )


    if elapsed > 0:

        speed = (

            remote_size

            / 1024**2

            / elapsed
        )

    else:

        speed = 0.0


    progress = (

        downloaded_bytes

        / total_bytes

        * 100.0

        if total_bytes > 0

        else 0.0
    )


    print(
        f"    done | "
        f"{elapsed:.1f}s | "
        f"{speed:.1f} MiB/s | "
        f"total progress "
        f"{progress:.2f}%"
    )


# ============================================================
# Complete
# ============================================================

elapsed_all = (
    time.time()
    - start_all
)


print()

print("=" * 80)

print(
    "DOWNLOAD COMPLETE"
)

print("=" * 80)


print(
    f"Downloaded size: "
    f"{downloaded_bytes / 1024**3:.2f} GiB"
)


print(
    f"Elapsed time: "
    f"{elapsed_all / 3600:.2f} hours"
)


print()


print(
    "Local dataset path:"
)


print(
    DST
)