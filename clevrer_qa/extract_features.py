from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
PHASE4_DIR = ROOT / "phase4"
VJEPA_REPO = ROOT / "vjepa2"
DATA_ROOT = ROOT / "data/clevrer"
CACHE_ROOT = ROOT / "clevrer_qa/cache"

OURS_CKPT = ROOT / "checkpoints/phase4/best.pt"
BASE_CKPT = ROOT / "checkpoints/phase4_baseline/best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SAMPLED_FRAMES = 48
CLIP_LEN = 16
NUM_CLIPS = NUM_SAMPLED_FRAMES // CLIP_LEN

LATENT_T = 8
GRID_H = 24
GRID_W = 24

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1, 1)

sys.path.insert(0, str(PHASE4_DIR))

from factorizer import WorldGridFactorizer
from common_leakfree import amp_context, extract_z


def decode_video(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )
        frames.append(frame)

    cap.release()

    if len(frames) < NUM_SAMPLED_FRAMES:
        raise RuntimeError(
            f"{path.name} has only {len(frames)} frames, "
            f"need at least {NUM_SAMPLED_FRAMES}"
        )

    return np.stack(frames, axis=0)


def sample_video(frames):
    indices = np.linspace(
        0,
        len(frames) - 1,
        NUM_SAMPLED_FRAMES,
    )

    indices = np.round(indices).astype(np.int64)

    return frames[indices]


def preprocess_clip(frames):
    x = torch.from_numpy(
        np.ascontiguousarray(frames)
    )

    x = x.permute(
        0,
        3,
        1,
        2,
    ).float() / 255.0

    x = F.interpolate(
        x,
        size=(384, 384),
        mode="bilinear",
        align_corners=False,
    )

    x = x.permute(
        1,
        0,
        2,
        3,
    ).unsqueeze(0)

    mean = IMAGE_MEAN.to(
        dtype=x.dtype,
        device=x.device,
    )

    std = IMAGE_STD.to(
        dtype=x.dtype,
        device=x.device,
    )

    x = (x - mean) / std

    return x


def load_model(representation):
    if representation == "raw":
        ckpt_path = BASE_CKPT
    else:
        ckpt_path = OURS_CKPT

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}"
        )

    print("Loading checkpoint:")
    print(ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE,
    )

    encoder, unused = torch.hub.load(
        str(VJEPA_REPO),
        "vjepa2_1_vit_base_384",
        source="local",
    )

    del unused

    encoder = encoder.to(DEVICE)

    missing, unexpected = encoder.load_state_dict(
        ckpt["encoder_trainable"],
        strict=False,
    )

    print(
        f"Encoder partial load: "
        f"{len(missing)} missing, "
        f"{len(unexpected)} unexpected"
    )

    encoder.eval()

    factorizer = None
    feature_dim = 768

    if representation == "world":
        world_dim = ckpt["config"]["world_dim"]
        grid_dim = ckpt["config"]["grid_dim"]

        factorizer = WorldGridFactorizer(
            input_dim=768,
            hidden_dim=768,
            world_dim=world_dim,
            grid_dim=grid_dim,
            num_grid_classes=4,
        ).to(DEVICE)

        factorizer.load_state_dict(
            ckpt["factorizer"]
        )

        factorizer.eval()

        feature_dim = world_dim

    return encoder, factorizer, feature_dim


def spatial_pool(z):
    """
    Accept either:

    [B, T, H, W, D]

    or

    [B, N, D]

    and return:

    [B, T, D]
    """

    if z.ndim == 5:
        b, t, h, w, d = z.shape

        if t != LATENT_T:
            raise RuntimeError(
                f"Unexpected temporal tokens: {t}, "
                f"expected {LATENT_T}"
            )

        if h != GRID_H or w != GRID_W:
            raise RuntimeError(
                f"Unexpected spatial grid: {h}x{w}, "
                f"expected {GRID_H}x{GRID_W}"
            )

    elif z.ndim == 3:
        b, n, d = z.shape

        expected = (
            LATENT_T
            * GRID_H
            * GRID_W
        )

        if n != expected:
            raise RuntimeError(
                f"Unexpected flattened token count: {n}, "
                f"expected {expected}"
            )

        z = z.reshape(
            b,
            LATENT_T,
            GRID_H,
            GRID_W,
            d,
        )

    else:
        raise RuntimeError(
            f"Unexpected latent shape: "
            f"{tuple(z.shape)}"
        )

    z = z.mean(
        dim=(2, 3)
    )

    return z


@torch.no_grad()
def encode_video(
    video_path,
    encoder,
    factorizer,
    representation,
):
    frames = decode_video(video_path)

    frames = sample_video(frames)

    all_steps = []

    for clip_idx in range(NUM_CLIPS):
        start = clip_idx * CLIP_LEN
        end = start + CLIP_LEN

        clip = frames[start:end]

        x = preprocess_clip(clip)
        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        with amp_context(DEVICE):
            z = extract_z(
                encoder,
                x,
                0,
            )

            if representation == "world":
                world, grid, logits = factorizer(z)
                z = world

        pooled = spatial_pool(z)

        all_steps.append(
            pooled.squeeze(0)
            .float()
            .cpu()
        )

    features = torch.cat(
        all_steps,
        dim=0,
    )

    expected_steps = (
        NUM_CLIPS
        * LATENT_T
    )

    if features.shape[0] != expected_steps:
        raise RuntimeError(
            f"Expected {expected_steps} temporal steps, "
            f"got {features.shape[0]}"
        )

    return features


def load_expected_videos(split):
    question_path = (
        DATA_ROOT
        / "questions"
        / f"{split}.json"
    )

    if not question_path.exists():
        raise FileNotFoundError(
            f"Missing question file: "
            f"{question_path}"
        )

    with question_path.open("r") as f:
        data = json.load(f)

    return [
        item["video_filename"]
        for item in data
    ]


def extract_split(
    split,
    representation,
    limit=None,
):
    video_dir = (
        DATA_ROOT
        / split
    )

    if not video_dir.exists():
        raise FileNotFoundError(
            f"Missing CLEVRER video directory:\n"
            f"{video_dir}"
        )

    names = load_expected_videos(
        split
    )

    if limit is not None:
        names = names[:limit]

    output_dir = (
        CACHE_ROOT
        / representation
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        output_dir
        / f"{split}_features.npy"
    )

    index_path = (
        output_dir
        / f"{split}_index.json"
    )

    meta_path = (
        output_dir
        / f"{split}_meta.json"
    )

    encoder, factorizer, feature_dim = load_model(
        representation
    )

    expected_steps = (
        NUM_CLIPS
        * LATENT_T
    )

    shape = (
        len(names),
        expected_steps,
        feature_dim,
    )

    print("=" * 100)
    print(
        f"CLEVRER FEATURE EXTRACTION: "
        f"{representation.upper()} / {split}"
    )
    print("=" * 100)

    print("Device:", DEVICE)
    print("Videos:", len(names))
    print("Output shape:", shape)
    print("Output:", feature_path)

    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )

    valid_names = []
    failed = []

    for i, name in enumerate(names):
        video_path = (
            video_dir
            / name
        )

        try:
            feat = encode_video(
                video_path,
                encoder,
                factorizer,
                representation,
            )

            write_idx = len(valid_names)

            features[write_idx] = (
                feat.numpy()
                .astype(np.float16)
            )

            valid_names.append(name)

            if (
                i % 50 == 0
                or i == len(names) - 1
            ):
                print(
                    f"Processed "
                    f"{i + 1}/{len(names)} | "
                    f"valid={len(valid_names)} | "
                    f"{name}"
                )

        except KeyboardInterrupt:
            print()
            print("Interrupted by user.")
            features.flush()
            raise

        except Exception as exc:
            failed.append(
                {
                    "video": name,
                    "error": str(exc),
                }
            )

            print(
                f"[skip] {name}: "
                f"{exc}"
            )

    features.flush()
    del features

    if len(valid_names) == 0:
        feature_path.unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "No valid videos were extracted."
        )

    if len(valid_names) != len(names):
        original = np.load(
            feature_path,
            mmap_mode="r",
        )

        compact_path = (
            output_dir
            / f"{split}_features_compact.npy"
        )

        compact = np.lib.format.open_memmap(
            compact_path,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(valid_names),
                expected_steps,
                feature_dim,
            ),
        )

        compact[:] = original[
            :len(valid_names)
        ]

        compact.flush()

        del compact
        del original

        feature_path.unlink()

        compact_path.rename(
            feature_path
        )

    index = {
        name: idx
        for idx, name
        in enumerate(valid_names)
    }

    with index_path.open("w") as f:
        json.dump(
            index,
            f,
            indent=2,
        )

    meta = {
        "representation": representation,
        "split": split,
        "num_requested": len(names),
        "num_valid": len(valid_names),
        "num_failed": len(failed),
        "feature_dim": feature_dim,
        "num_temporal_steps": expected_steps,
        "num_sampled_frames": NUM_SAMPLED_FRAMES,
        "clip_len": CLIP_LEN,
        "num_clips": NUM_CLIPS,
        "latent_t": LATENT_T,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "sampling": (
            "Uniformly sample 48 frames "
            "over the full CLEVRER video, "
            "then split into three "
            "non-overlapping 16-frame clips."
        ),
        "pooling": (
            "Mean pool each latent temporal step "
            "over the 24x24 spatial grid."
        ),
        "checkpoint": str(
            BASE_CKPT
            if representation == "raw"
            else OURS_CKPT
        ),
        "failed": failed,
    }

    with meta_path.open("w") as f:
        json.dump(
            meta,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        "Requested videos:",
        len(names),
    )

    print(
        "Valid videos:",
        len(valid_names),
    )

    print(
        "Failed videos:",
        len(failed),
    )

    print(
        "Features:",
        feature_path,
    )

    print(
        "Index:",
        index_path,
    )

    print(
        "Metadata:",
        meta_path,
    )

    if failed:
        print()
        print("First failed examples:")

        for item in failed[:10]:
            print(
                item["video"],
                "->",
                item["error"],
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--representation",
        choices=[
            "raw",
            "world",
        ],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
        ],
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional debugging limit. "
            "Do not use for final experiments."
        ),
    )

    args = parser.parse_args()

    extract_split(
        split=args.split,
        representation=args.representation,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()