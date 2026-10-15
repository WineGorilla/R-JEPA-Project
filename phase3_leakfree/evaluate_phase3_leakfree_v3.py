from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch

from factorizer import WorldGridFactorizer
from temporal_predictor import TemporalPredictor, multi_horizon_loss

from common_leakfree import (
    amp_context,
    prepare_past_future,
    extract_z,
    temporal_pair,
    copy_last_loss,
    relative_gain,
    cosine_delta,
)

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VJEPA_REPO = ROOT / "vjepa2"
DATA_DIR = ROOT / "data/tfds"

OURS_CKPT = ROOT / "checkpoints/phase3_leakfree_v3/best.pt"
BASE_CKPT = ROOT / "checkpoints/phase3_baseline_leakfree/best.pt"

NUM_TEST_CLIPS = 128

TEST_PHASES = [
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]

TRAINED_PHASES = {
    0,
    4,
    8,
    12,
}

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

ours_encoder.load_state_dict(
    ours_ckpt["encoder_trainable"],
    strict=False,
)

baseline_encoder, unused = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)

del unused

baseline_encoder = baseline_encoder.to(DEVICE)

baseline_encoder.load_state_dict(
    base_ckpt["encoder_trainable"],
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

world_predictor = TemporalPredictor(
    input_dim=WORLD_DIM,
    output_dim=WORLD_DIM,
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
        "base_pred": [],
        "base_copy": [],
        "ours_pred": [],
        "ours_copy": [],
        "base_delta": [],
        "world_delta": [],
    }

print("=" * 100)
print("LEAKAGE-FREE PHASE-3 TEST")
print("=" * 100)

with torch.no_grad():
    for clip_idx, sample in enumerate(dataset):
        past, future = prepare_past_future(sample)

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
                bp = extract_z(
                    baseline_encoder,
                    past,
                    phase,
                )

                bf = extract_z(
                    baseline_encoder,
                    future,
                    phase,
                )

                bc, bt = temporal_pair(
                    bp,
                    bf,
                )

                bpred = raw_predictor(bc)

                bloss = multi_horizon_loss(
                    bpred,
                    bt,
                )["total"]

                bcopy = copy_last_loss(
                    bc,
                    bt,
                )["total"]

                op = extract_z(
                    ours_encoder,
                    past,
                    phase,
                )

                of = extract_z(
                    ours_encoder,
                    future,
                    phase,
                )

                wp, _, _ = factorizer(op)
                wf, _, _ = factorizer(of)

                oc, ot = temporal_pair(
                    wp,
                    wf,
                )

                opred = world_predictor(oc)

                oloss = multi_horizon_loss(
                    opred,
                    ot,
                )["total"]

                ocopy = copy_last_loss(
                    oc,
                    ot,
                )["total"]

                bdelta = cosine_delta(
                    base_ref,
                    bp,
                )

                wdelta = cosine_delta(
                    ours_ref_w,
                    wp,
                )

            r = results[phase]

            r["base_pred"].append(float(bloss))
            r["base_copy"].append(float(bcopy))
            r["ours_pred"].append(float(oloss))
            r["ours_copy"].append(float(ocopy))
            r["base_delta"].append(float(bdelta))
            r["world_delta"].append(float(wdelta))

        if clip_idx % 8 == 0:
            print(f"Finished clip {clip_idx}")

summary = {}

print()
print("=" * 110)
print("RESULTS")
print("=" * 110)

print(
    f"{'Phase':>7} | "
    f"{'Type':>6} | "
    f"{'Base Gain':>10} | "
    f"{'Ours Gain':>10} | "
    f"{'Advantage':>10} | "
    f"{'Base Δ':>8} | "
    f"{'World Δ':>8}"
)

print("-" * 110)

for phase in TEST_PHASES:
    r = results[phase]

    bp = float(np.mean(r["base_pred"]))
    bc = float(np.mean(r["base_copy"]))

    op = float(np.mean(r["ours_pred"]))
    oc = float(np.mean(r["ours_copy"]))

    bg = relative_gain(bp, bc)
    og = relative_gain(op, oc)

    advantage = (og - bg) * 100.0

    bd = float(np.mean(r["base_delta"]))
    wd = float(np.mean(r["world_delta"]))

    summary[phase] = {
        "base_gain": bg,
        "ours_gain": og,
        "advantage": advantage,
        "base_delta": bd,
        "world_delta": wd,
    }

    phase_type = (
        "seen"
        if phase in TRAINED_PHASES
        else "unseen"
    )

    print(
        f"{phase:>5}px | "
        f"{phase_type:>6} | "
        f"{bg * 100:>9.2f}% | "
        f"{og * 100:>9.2f}% | "
        f"{advantage:>+8.2f}pp | "
        f"{bd:>8.4f} | "
        f"{wd:>8.4f}"
    )

seen = [
    p
    for p in TEST_PHASES
    if p in TRAINED_PHASES
]

unseen = [
    p
    for p in TEST_PHASES
    if p not in TRAINED_PHASES
]


def avg(phases, key):
    return float(
        np.mean(
            [summary[p][key] for p in phases]
        )
    )


print()
print("=" * 110)
print("SEEN")
print("=" * 110)

print(
    f"Baseline gain : "
    f"{avg(seen, 'base_gain') * 100:.2f}%"
)

print(
    f"Ours gain     : "
    f"{avg(seen, 'ours_gain') * 100:.2f}%"
)

print(
    f"Advantage     : "
    f"{avg(seen, 'advantage'):+.2f} pp"
)

print()
print("=" * 110)
print("UNSEEN")
print("=" * 110)

print(
    f"Baseline gain : "
    f"{avg(unseen, 'base_gain') * 100:.2f}%"
)

print(
    f"Ours gain     : "
    f"{avg(unseen, 'ours_gain') * 100:.2f}%"
)

print(
    f"Advantage     : "
    f"{avg(unseen, 'advantage'):+.2f} pp"
)

nonzero = [
    p
    for p in TEST_PHASES
    if p != 0
]

base_delta = avg(
    nonzero,
    "base_delta",
)

world_delta = avg(
    nonzero,
    "world_delta",
)

reduction = (
    1.0
    - world_delta / base_delta
) * 100.0

print()
print("=" * 110)
print("PHASE SENSITIVITY")
print("=" * 110)

print(
    f"Baseline Raw mean Δ : "
    f"{base_delta:.6f}"
)

print(
    f"Ours World mean Δ   : "
    f"{world_delta:.6f}"
)

print(
    f"Reduction            : "
    f"{reduction:.2f}%"
)

print()
print("DONE")