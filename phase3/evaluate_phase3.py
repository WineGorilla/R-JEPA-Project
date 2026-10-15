from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch

from factorizer import WorldGridFactorizer
from temporal_predictor import TemporalPredictor, multi_horizon_loss
from common import (
    amp_context,
    prepare_video,
    extract_z,
    split_temporal,
    copy_last_loss,
    relative_gain,
    cosine_delta,
)

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VJEPA_REPO = ROOT / "vjepa2"
DATA_DIR = ROOT / "data/tfds"

OURS_CKPT = ROOT / "checkpoints/phase3/best.pt"
BASELINE_CKPT = ROOT / "checkpoints/phase3_baseline/best.pt"

NUM_TEST_CLIPS = 128
TEST_PHASES = [0, 2, 4, 6, 8, 10, 12, 14, 16]
TRAINED_PHASES = {0, 4, 8, 12}

ours_ckpt = torch.load(
    OURS_CKPT,
    map_location=DEVICE,
)

baseline_ckpt = torch.load(
    BASELINE_CKPT,
    map_location=DEVICE,
)

print("=" * 80)
print("Loading Phase-3 full model")
print("=" * 80)

ours_encoder, unused = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)
del unused
ours_encoder = ours_encoder.to(DEVICE)

ours_encoder.load_state_dict(
    ours_ckpt["encoder_trainable"],
    strict=False,
)

WORLD_DIM = ours_ckpt["config"]["world_dim"]
GRID_DIM = ours_ckpt["config"]["grid_dim"]

factorizer = WorldGridFactorizer(
    input_dim=768,
    hidden_dim=768,
    world_dim=WORLD_DIM,
    grid_dim=GRID_DIM,
    num_grid_classes=4,
).to(DEVICE)

factorizer.load_state_dict(
    ours_ckpt["factorizer"]
)

phase2 = torch.load(
    ROOT / "checkpoints/phase2/best.pt",
    map_location=DEVICE,
)

world_predictor = TemporalPredictor(
    input_dim=WORLD_DIM,
    output_dim=WORLD_DIM,
    hidden_dim=phase2["config"]["hidden_dim"],
    num_layers=phase2["config"]["layers"],
    num_heads=phase2["config"]["heads"],
    num_horizons=len(phase2["config"]["target_indices"]),
).to(DEVICE)

world_predictor.load_state_dict(
    ours_ckpt["world_predictor"]
)

print("=" * 80)
print("Loading matched baseline")
print("=" * 80)

baseline_encoder, unused = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)
del unused
baseline_encoder = baseline_encoder.to(DEVICE)

baseline_encoder.load_state_dict(
    baseline_ckpt["encoder_trainable"],
    strict=False,
)

raw_predictor = TemporalPredictor(
    input_dim=768,
    output_dim=768,
    hidden_dim=phase2["config"]["hidden_dim"],
    num_layers=phase2["config"]["layers"],
    num_heads=phase2["config"]["heads"],
    num_horizons=len(phase2["config"]["target_indices"]),
).to(DEVICE)

raw_predictor.load_state_dict(
    baseline_ckpt["raw_predictor"]
)

ours_encoder.eval()
factorizer.eval()
world_predictor.eval()
baseline_encoder.eval()
raw_predictor.eval()

builder = tfds.builder(
    "movi_c/256x256",
    data_dir=str(DATA_DIR),
)

dataset = builder.as_dataset(
    split=f"test[:{NUM_TEST_CLIPS}]",
    shuffle_files=False,
)

results = {}

for phase in TEST_PHASES:
    results[phase] = {
        "baseline_pred": [],
        "baseline_copy": [],
        "ours_pred": [],
        "ours_copy": [],
        "baseline_phase_delta": [],
        "ours_phase_delta": [],
    }

print()
print("=" * 80)
print("Evaluating Phase-3 on MOVi-C test split")
print("=" * 80)

with torch.no_grad():
    for clip_idx, sample in enumerate(dataset):
        video = prepare_video(sample).to(DEVICE)

        with amp_context(DEVICE):
            baseline_ref = extract_z(
                baseline_encoder,
                video,
                0,
            )

            ours_z_ref = extract_z(
                ours_encoder,
                video,
                0,
            )

            ours_world_ref, _, _ = factorizer(
                ours_z_ref
            )

        for phase in TEST_PHASES:
            with amp_context(DEVICE):
                baseline_z = extract_z(
                    baseline_encoder,
                    video,
                    phase,
                )

                baseline_context, baseline_targets = split_temporal(
                    baseline_z
                )

                baseline_predictions = raw_predictor(
                    baseline_context
                )

                baseline_pred = multi_horizon_loss(
                    baseline_predictions,
                    baseline_targets,
                )["total"]

                baseline_copy = copy_last_loss(
                    baseline_context,
                    baseline_targets,
                )["total"]

                ours_z = extract_z(
                    ours_encoder,
                    video,
                    phase,
                )

                ours_world, _, _ = factorizer(
                    ours_z
                )

                ours_context, ours_targets = split_temporal(
                    ours_world
                )

                ours_predictions = world_predictor(
                    ours_context
                )

                ours_pred = multi_horizon_loss(
                    ours_predictions,
                    ours_targets,
                )["total"]

                ours_copy = copy_last_loss(
                    ours_context,
                    ours_targets,
                )["total"]

                baseline_delta = cosine_delta(
                    baseline_ref,
                    baseline_z,
                )

                ours_delta = cosine_delta(
                    ours_world_ref,
                    ours_world,
                )

            r = results[phase]

            r["baseline_pred"].append(
                float(baseline_pred)
            )

            r["baseline_copy"].append(
                float(baseline_copy)
            )

            r["ours_pred"].append(
                float(ours_pred)
            )

            r["ours_copy"].append(
                float(ours_copy)
            )

            r["baseline_phase_delta"].append(
                float(baseline_delta)
            )

            r["ours_phase_delta"].append(
                float(ours_delta)
            )

        if clip_idx % 8 == 0:
            print(
                f"Finished clip {clip_idx}"
            )

summary = {}

print()
print("=" * 105)
print("PHASE-3 TEST RESULTS")
print("=" * 105)

print(
    f"{'Phase':>7} | "
    f"{'Type':>6} | "
    f"{'Baseline Gain':>13} | "
    f"{'Ours Gain':>10} | "
    f"{'Advantage':>10} | "
    f"{'Base Δ':>8} | "
    f"{'World Δ':>8}"
)

print("-" * 105)

for phase in TEST_PHASES:
    r = results[phase]

    baseline_pred = float(
        np.mean(r["baseline_pred"])
    )

    baseline_copy = float(
        np.mean(r["baseline_copy"])
    )

    ours_pred = float(
        np.mean(r["ours_pred"])
    )

    ours_copy = float(
        np.mean(r["ours_copy"])
    )

    baseline_gain = relative_gain(
        baseline_pred,
        baseline_copy,
    )

    ours_gain = relative_gain(
        ours_pred,
        ours_copy,
    )

    advantage = (
        ours_gain
        - baseline_gain
    ) * 100.0

    baseline_delta = float(
        np.mean(r["baseline_phase_delta"])
    )

    ours_delta = float(
        np.mean(r["ours_phase_delta"])
    )

    summary[phase] = {
        "baseline_gain": baseline_gain,
        "ours_gain": ours_gain,
        "advantage": advantage,
        "baseline_delta": baseline_delta,
        "ours_delta": ours_delta,
    }

    phase_type = (
        "seen"
        if phase in TRAINED_PHASES
        else "unseen"
    )

    print(
        f"{phase:>5}px | "
        f"{phase_type:>6} | "
        f"{baseline_gain * 100:>12.2f}% | "
        f"{ours_gain * 100:>9.2f}% | "
        f"{advantage:>+8.2f}pp | "
        f"{baseline_delta:>8.4f} | "
        f"{ours_delta:>8.4f}"
    )

seen = [
    phase
    for phase in TEST_PHASES
    if phase in TRAINED_PHASES
]

unseen = [
    phase
    for phase in TEST_PHASES
    if phase not in TRAINED_PHASES
]


def average(phases, key):
    return float(
        np.mean(
            [
                summary[p][key]
                for p in phases
            ]
        )
    )


print()
print("=" * 105)
print("SEEN PHASES")
print("=" * 105)

print(
    f"Baseline mean gain : "
    f"{average(seen, 'baseline_gain') * 100:.2f}%"
)

print(
    f"Ours mean gain     : "
    f"{average(seen, 'ours_gain') * 100:.2f}%"
)

print(
    f"Ours advantage     : "
    f"{average(seen, 'advantage'):+.2f} pp"
)

print()
print("=" * 105)
print("UNSEEN PHASES")
print("=" * 105)

print(
    f"Baseline mean gain : "
    f"{average(unseen, 'baseline_gain') * 100:.2f}%"
)

print(
    f"Ours mean gain     : "
    f"{average(unseen, 'ours_gain') * 100:.2f}%"
)

print(
    f"Ours advantage     : "
    f"{average(unseen, 'advantage'):+.2f} pp"
)

print()
print("=" * 105)
print("PHASE SENSITIVITY")
print("=" * 105)

nonzero = [
    phase
    for phase in TEST_PHASES
    if phase != 0
]

print(
    f"Baseline Raw mean Δ : "
    f"{average(nonzero, 'baseline_delta'):.6f}"
)

print(
    f"Ours World mean Δ   : "
    f"{average(nonzero, 'ours_delta'):.6f}"
)

reduction = (
    1.0
    - average(nonzero, "ours_delta")
    / average(nonzero, "baseline_delta")
) * 100.0

print(
    f"World reduction     : "
    f"{reduction:.2f}%"
)

print()
print("=" * 105)
print("DONE")
print("=" * 105)