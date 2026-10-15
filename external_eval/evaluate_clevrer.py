from pathlib import Path
import csv
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PHASE4_DIR = ROOT / "phase4"
sys.path.insert(0, str(PHASE4_DIR))

from factorizer import WorldGridFactorizer
from temporal_predictor import TemporalPredictor, multi_horizon_loss
from common_leakfree import (
    amp_context,
    extract_z,
    temporal_pair,
    copy_last_loss,
    relative_gain,
    cosine_delta,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VJEPA_REPO = ROOT / "vjepa2"
CLEVRER_ROOT = ROOT / "data/clevrer"

OURS_CKPT = ROOT / "checkpoints/phase4/best.pt"
BASE_CKPT = ROOT / "checkpoints/phase4_baseline/best.pt"

RESULT_DIR = ROOT / "external_eval/results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "CLEVRER"
VIDEO_DIR = CLEVRER_ROOT / "validation"
NUM_TEST_CLIPS = 128

TEST_PHASES = [0, 2, 4, 6, 8, 10, 12, 14, 16]
TRAINED_PHASES = {0, 4, 8, 12}

IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1, 1)


def load_models():
    if not OURS_CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {OURS_CKPT}")

    if not BASE_CKPT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {BASE_CKPT}")

    ours_ckpt = torch.load(
        OURS_CKPT,
        map_location=DEVICE,
    )

    base_ckpt = torch.load(
        BASE_CKPT,
        map_location=DEVICE,
    )

    ours_encoder, unused = torch.hub.load(
        str(VJEPA_REPO),
        "vjepa2_1_vit_base_384",
        source="local",
    )
    del unused

    ours_encoder = ours_encoder.to(DEVICE)

    missing, unexpected = ours_encoder.load_state_dict(
        ours_ckpt["encoder_trainable"],
        strict=False,
    )

    print(
        f"Ours encoder partial load: "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )

    baseline_encoder, unused = torch.hub.load(
        str(VJEPA_REPO),
        "vjepa2_1_vit_base_384",
        source="local",
    )
    del unused

    baseline_encoder = baseline_encoder.to(DEVICE)

    missing, unexpected = baseline_encoder.load_state_dict(
        base_ckpt["encoder_trainable"],
        strict=False,
    )

    print(
        f"Baseline encoder partial load: "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )

    world_dim = ours_ckpt["config"]["world_dim"]
    grid_dim = ours_ckpt["config"]["grid_dim"]

    factorizer = WorldGridFactorizer(
        input_dim=768,
        hidden_dim=768,
        world_dim=world_dim,
        grid_dim=grid_dim,
        num_grid_classes=4,
    ).to(DEVICE)

    factorizer.load_state_dict(
        ours_ckpt["factorizer"]
    )

    world_predictor = TemporalPredictor(
        input_dim=world_dim,
        output_dim=world_dim,
        hidden_dim=384,
        num_layers=3,
        num_heads=6,
        num_horizons=3,
    ).to(DEVICE)

    world_predictor.load_state_dict(
        ours_ckpt["world_predictor"]
    )

    raw_predictor = TemporalPredictor(
        input_dim=768,
        output_dim=768,
        hidden_dim=384,
        num_layers=3,
        num_heads=6,
        num_horizons=3,
    ).to(DEVICE)

    raw_predictor.load_state_dict(
        base_ckpt["raw_predictor"]
    )

    ours_encoder.eval()
    baseline_encoder.eval()
    factorizer.eval()
    world_predictor.eval()
    raw_predictor.eval()

    return (
        ours_encoder,
        baseline_encoder,
        factorizer,
        world_predictor,
        raw_predictor,
    )


def find_videos():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"CLEVRER video directory not found: {VIDEO_DIR}"
        )

    videos = sorted(
        VIDEO_DIR.glob("*.mp4")
    )

    if not videos:
        raise FileNotFoundError(
            f"No mp4 videos found in: {VIDEO_DIR}"
        )

    videos = videos[:NUM_TEST_CLIPS]

    print(
        f"Found {len(videos)} CLEVRER videos."
    )

    return videos


def decode_video(path):
    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}"
        )

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

    if len(frames) < 14:
        raise RuntimeError(
            f"Only {len(frames)} frames: {path.name}"
        )

    return np.stack(
        frames,
        axis=0,
    )


def choose_14_frames(frames):
    if len(frames) == 14:
        return frames

    indices = np.linspace(
        0,
        len(frames) - 1,
        14,
    ).round().astype(np.int64)

    return frames[indices]


def preprocess_video(frames):
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

    x = (
        x - mean
    ) / std

    return x


def prepare_leakage_free_pair(frames):
    frames = choose_14_frames(frames)

    past_true = frames[:8]
    future_true = frames[8:14]

    past_pad = np.repeat(
        past_true[-1:],
        8,
        axis=0,
    )

    future_pad = np.repeat(
        future_true[-1:],
        10,
        axis=0,
    )

    past_frames = np.concatenate(
        [
            past_true,
            past_pad,
        ],
        axis=0,
    )

    future_frames = np.concatenate(
        [
            future_true,
            future_pad,
        ],
        axis=0,
    )

    past = preprocess_video(
        past_frames
    )

    future = preprocess_video(
        future_frames
    )

    return past, future


def mean_for(summary, phases, key):
    return float(
        np.mean([summary[p][key] for p in phases])
    )


def save_results(summary, aggregate, valid_clips):
    csv_path = RESULT_DIR / "clevrer_phase4.csv"
    json_path = RESULT_DIR / "clevrer_phase4.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "dataset",
                "phase_px",
                "phase_type",
                "baseline_gain_pct",
                "ours_gain_pct",
                "advantage_pp",
                "baseline_delta",
                "world_delta",
            ]
        )

        for phase in TEST_PHASES:
            r = summary[phase]

            writer.writerow(
                [
                    DATASET_NAME,
                    phase,
                    (
                        "seen"
                        if phase in TRAINED_PHASES
                        else "unseen"
                    ),
                    r["base_gain"] * 100.0,
                    r["ours_gain"] * 100.0,
                    r["advantage"],
                    r["base_delta"],
                    r["world_delta"],
                ]
            )

    payload = {
        "dataset": DATASET_NAME,
        "video_dir": str(VIDEO_DIR),
        "requested_clips": NUM_TEST_CLIPS,
        "valid_clips": valid_clips,
        "trained_offsets": sorted(TRAINED_PHASES),
        "test_offsets": TEST_PHASES,
        "frame_sampling": "14 uniformly sampled frames over each video",
        "per_phase": {
            str(k): v
            for k, v in summary.items()
        },
        "aggregate": aggregate,
    }

    with json_path.open("w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    print()
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


def summarize(results, valid_clips):
    summary = {}

    print()
    print("=" * 112)
    print("RESULTS")
    print("=" * 112)

    print(
        f"{'Phase':>7} | "
        f"{'Type':>6} | "
        f"{'Base Gain':>10} | "
        f"{'Ours Gain':>10} | "
        f"{'Advantage':>10} | "
        f"{'Base Δ':>10} | "
        f"{'World Δ':>10}"
    )

    print("-" * 112)

    for phase in TEST_PHASES:
        r = results[phase]

        base_pred = float(
            np.mean(r["base_pred"])
        )

        base_copy = float(
            np.mean(r["base_copy"])
        )

        ours_pred = float(
            np.mean(r["ours_pred"])
        )

        ours_copy = float(
            np.mean(r["ours_copy"])
        )

        base_gain = relative_gain(
            base_pred,
            base_copy,
        )

        ours_gain = relative_gain(
            ours_pred,
            ours_copy,
        )

        advantage = (
            ours_gain - base_gain
        ) * 100.0

        base_delta = float(
            np.mean(r["base_delta"])
        )

        world_delta = float(
            np.mean(r["world_delta"])
        )

        summary[phase] = {
            "base_gain": base_gain,
            "ours_gain": ours_gain,
            "advantage": advantage,
            "base_delta": base_delta,
            "world_delta": world_delta,
        }

        phase_type = (
            "seen"
            if phase in TRAINED_PHASES
            else "unseen"
        )

        print(
            f"{phase:>5}px | "
            f"{phase_type:>6} | "
            f"{base_gain * 100:>9.2f}% | "
            f"{ours_gain * 100:>9.2f}% | "
            f"{advantage:>+8.2f}pp | "
            f"{base_delta:>10.6f} | "
            f"{world_delta:>10.6f}"
        )

    seen = [
        p for p in TEST_PHASES
        if p in TRAINED_PHASES
    ]

    unseen = [
        p for p in TEST_PHASES
        if p not in TRAINED_PHASES
    ]

    nonzero = [
        p for p in TEST_PHASES
        if p != 0
    ]

    seen_base = mean_for(
        summary,
        seen,
        "base_gain",
    )

    seen_ours = mean_for(
        summary,
        seen,
        "ours_gain",
    )

    unseen_base = mean_for(
        summary,
        unseen,
        "base_gain",
    )

    unseen_ours = mean_for(
        summary,
        unseen,
        "ours_gain",
    )

    overall_base = mean_for(
        summary,
        TEST_PHASES,
        "base_gain",
    )

    overall_ours = mean_for(
        summary,
        TEST_PHASES,
        "ours_gain",
    )

    base_delta = mean_for(
        summary,
        nonzero,
        "base_delta",
    )

    world_delta = mean_for(
        summary,
        nonzero,
        "world_delta",
    )

    reduction = (
        1.0 - world_delta / base_delta
    ) * 100.0

    aggregate = {
        "seen_baseline_gain_pct": seen_base * 100.0,
        "seen_ours_gain_pct": seen_ours * 100.0,
        "seen_advantage_pp": (
            seen_ours - seen_base
        ) * 100.0,
        "unseen_baseline_gain_pct": unseen_base * 100.0,
        "unseen_ours_gain_pct": unseen_ours * 100.0,
        "unseen_advantage_pp": (
            unseen_ours - unseen_base
        ) * 100.0,
        "overall_baseline_gain_pct": overall_base * 100.0,
        "overall_ours_gain_pct": overall_ours * 100.0,
        "overall_advantage_pp": (
            overall_ours - overall_base
        ) * 100.0,
        "baseline_phase_delta": base_delta,
        "world_phase_delta": world_delta,
        "phase_reduction_pct": reduction,
    }

    print()
    print("=" * 112)
    print("AGGREGATE")
    print("=" * 112)

    print(
        f"Valid clips          : "
        f"{valid_clips}"
    )

    print(
        f"Seen baseline gain   : "
        f"{aggregate['seen_baseline_gain_pct']:.2f}%"
    )

    print(
        f"Seen ours gain       : "
        f"{aggregate['seen_ours_gain_pct']:.2f}%"
    )

    print(
        f"Seen advantage       : "
        f"{aggregate['seen_advantage_pp']:+.2f} pp"
    )

    print(
        f"Unseen baseline gain : "
        f"{aggregate['unseen_baseline_gain_pct']:.2f}%"
    )

    print(
        f"Unseen ours gain     : "
        f"{aggregate['unseen_ours_gain_pct']:.2f}%"
    )

    print(
        f"Unseen advantage     : "
        f"{aggregate['unseen_advantage_pp']:+.2f} pp"
    )

    print(
        f"Overall baseline gain: "
        f"{aggregate['overall_baseline_gain_pct']:.2f}%"
    )

    print(
        f"Overall ours gain    : "
        f"{aggregate['overall_ours_gain_pct']:.2f}%"
    )

    print(
        f"Overall advantage    : "
        f"{aggregate['overall_advantage_pp']:+.2f} pp"
    )

    print(
        f"Baseline Raw mean Δ  : "
        f"{base_delta:.6f}"
    )

    print(
        f"Ours World mean Δ    : "
        f"{world_delta:.6f}"
    )

    print(
        f"Phase reduction      : "
        f"{reduction:.2f}%"
    )

    save_results(
        summary,
        aggregate,
        valid_clips,
    )


def main():
    print("=" * 100)
    print(f"PHASE-4 EXTERNAL EVALUATION: {DATASET_NAME}")
    print("=" * 100)

    print("Device:", DEVICE)
    print("Video directory:", VIDEO_DIR)
    print("Ours checkpoint:", OURS_CKPT)
    print("Baseline checkpoint:", BASE_CKPT)

    (
        ours_encoder,
        baseline_encoder,
        factorizer,
        world_predictor,
        raw_predictor,
    ) = load_models()

    videos = find_videos()

    results = {
        phase: {
            "base_pred": [],
            "base_copy": [],
            "ours_pred": [],
            "ours_copy": [],
            "base_delta": [],
            "world_delta": [],
        }
        for phase in TEST_PHASES
    }

    valid_clips = 0

    with torch.no_grad():
        for clip_idx, video_path in enumerate(videos):
            try:
                frames = decode_video(
                    video_path
                )

                past, future = prepare_leakage_free_pair(
                    frames
                )

                past = past.to(DEVICE)
                future = future.to(DEVICE)

                with amp_context(DEVICE):
                    base_ref = extract_z(
                        baseline_encoder,
                        past,
                        0,
                    )

                    ours_ref_z = extract_z(
                        ours_encoder,
                        past,
                        0,
                    )

                    ours_ref_w, _, _ = factorizer(
                        ours_ref_z
                    )

                for phase in TEST_PHASES:
                    with amp_context(DEVICE):
                        base_past = extract_z(
                            baseline_encoder,
                            past,
                            phase,
                        )

                        base_future = extract_z(
                            baseline_encoder,
                            future,
                            phase,
                        )

                        base_context, base_targets = temporal_pair(
                            base_past,
                            base_future,
                        )

                        base_predictions = raw_predictor(
                            base_context
                        )

                        base_loss = multi_horizon_loss(
                            base_predictions,
                            base_targets,
                        )["total"]

                        base_copy = copy_last_loss(
                            base_context,
                            base_targets,
                        )["total"]

                        ours_past_z = extract_z(
                            ours_encoder,
                            past,
                            phase,
                        )

                        ours_future_z = extract_z(
                            ours_encoder,
                            future,
                            phase,
                        )

                        ours_past_w, _, _ = factorizer(
                            ours_past_z
                        )

                        ours_future_w, _, _ = factorizer(
                            ours_future_z
                        )

                        ours_context, ours_targets = temporal_pair(
                            ours_past_w,
                            ours_future_w,
                        )

                        ours_predictions = world_predictor(
                            ours_context
                        )

                        ours_loss = multi_horizon_loss(
                            ours_predictions,
                            ours_targets,
                        )["total"]

                        ours_copy = copy_last_loss(
                            ours_context,
                            ours_targets,
                        )["total"]

                        base_delta = cosine_delta(
                            base_ref,
                            base_past,
                        )

                        world_delta = cosine_delta(
                            ours_ref_w,
                            ours_past_w,
                        )

                    r = results[phase]

                    r["base_pred"].append(
                        float(base_loss)
                    )

                    r["base_copy"].append(
                        float(base_copy)
                    )

                    r["ours_pred"].append(
                        float(ours_loss)
                    )

                    r["ours_copy"].append(
                        float(ours_copy)
                    )

                    r["base_delta"].append(
                        float(base_delta)
                    )

                    r["world_delta"].append(
                        float(world_delta)
                    )

                valid_clips += 1

                if clip_idx % 8 == 0:
                    print(
                        f"Finished clip "
                        f"{clip_idx}/{len(videos)}: "
                        f"{video_path.name}"
                    )

            except Exception as exc:
                print(
                    f"Skipping {video_path.name}: "
                    f"{exc}"
                )

    if valid_clips == 0:
        raise RuntimeError(
            "No CLEVRER videos were evaluated successfully."
        )

    summarize(
        results,
        valid_clips,
    )


if __name__ == "__main__":
    main()
