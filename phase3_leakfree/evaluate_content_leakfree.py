from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F

from factorizer import WorldGridFactorizer

from common_leakfree import (
    amp_context,
    prepare_past_future,
    extract_z,
    cosine_delta,
)

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VJEPA_REPO = ROOT / "vjepa2"
DATA_DIR = ROOT / "data/tfds"

OURS_CKPT = ROOT / "checkpoints/phase3_leakfree/best.pt"

NUM_CLIPS = 64
PHASE = 8

ckpt = torch.load(
    OURS_CKPT,
    map_location=DEVICE,
)

encoder, unused = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)

del unused

encoder = encoder.to(DEVICE)

encoder.load_state_dict(
    ckpt["encoder_trainable"],
    strict=False,
)

WORLD_DIM = ckpt["config"]["world_dim"]
GRID_DIM = ckpt["config"]["grid_dim"]

factorizer = WorldGridFactorizer(
    input_dim=768,
    hidden_dim=768,
    world_dim=WORLD_DIM,
    grid_dim=GRID_DIM,
    num_grid_classes=4,
).to(DEVICE)

factorizer.load_state_dict(
    ckpt["factorizer"]
)

encoder.eval()
factorizer.eval()

builder = tfds.builder(
    "movi_c/256x256",
    data_dir=str(DATA_DIR),
)

dataset = builder.as_dataset(
    split=f"test[:{NUM_CLIPS}]",
    shuffle_files=False,
)

world0_all = []
world8_all = []

token_phase_deltas = []

print("=" * 80)
print("CONTENT PRESERVATION / COLLAPSE CHECK")
print("=" * 80)

with torch.no_grad():
    for idx, sample in enumerate(dataset):
        past, _ = prepare_past_future(sample)
        past = past.to(DEVICE)

        with amp_context(DEVICE):
            z0 = extract_z(
                encoder,
                past,
                0,
            )

            z8 = extract_z(
                encoder,
                past,
                PHASE,
            )

            w0, _, _ = factorizer(z0)
            w8, _, _ = factorizer(z8)

        delta = cosine_delta(
            w0,
            w8,
        )

        token_phase_deltas.append(
            float(delta)
        )

        pooled0 = w0.float().mean(
            dim=(1, 2, 3)
        )

        pooled8 = w8.float().mean(
            dim=(1, 2, 3)
        )

        pooled0 = F.normalize(
            pooled0,
            dim=-1,
        )

        pooled8 = F.normalize(
            pooled8,
            dim=-1,
        )

        world0_all.append(
            pooled0.cpu()
        )

        world8_all.append(
            pooled8.cpu()
        )

        if idx % 8 == 0:
            print(
                f"Finished clip {idx}"
            )

world0 = torch.cat(
    world0_all,
    dim=0,
)

world8 = torch.cat(
    world8_all,
    dim=0,
)

same_video_phase = (
    1.0
    - (world0 * world8).sum(dim=-1)
)

different_video = []

n = world0.shape[0]

for i in range(n):
    for j in range(i + 1, n):
        d = 1.0 - torch.dot(
            world0[i],
            world0[j],
        )

        different_video.append(
            float(d)
        )

different_video = np.array(
    different_video
)

same_video_phase = (
    same_video_phase
    .numpy()
)

token_phase = np.array(
    token_phase_deltas
)

world_matrix = torch.cat(
    [world0, world8],
    dim=0,
)

feature_std = world_matrix.std(
    dim=0,
    unbiased=False,
).mean().item()

centered = (
    world_matrix
    - world_matrix.mean(
        dim=0,
        keepdim=True,
    )
)

singular_values = torch.linalg.svdvals(
    centered
)

variance = singular_values ** 2
variance = variance / variance.sum()

effective_rank = torch.exp(
    -(
        variance
        * torch.log(
            variance + 1e-12
        )
    ).sum()
).item()

mean_same = float(
    same_video_phase.mean()
)

mean_diff = float(
    different_video.mean()
)

separation = (
    mean_diff
    / max(
        mean_same,
        1e-8,
    )
)

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)

print(
    f"Token-level phase Δ      : "
    f"{token_phase.mean():.6f}"
)

print(
    f"Global same-video phase  : "
    f"{mean_same:.6f}"
)

print(
    f"Different-video distance : "
    f"{mean_diff:.6f}"
)

print(
    f"Separation ratio         : "
    f"{separation:.2f}x"
)

print(
    f"Mean feature std         : "
    f"{feature_std:.6f}"
)

print(
    f"Effective rank           : "
    f"{effective_rank:.2f}"
)

print()
print("=" * 80)
print("DIAGNOSTIC")
print("=" * 80)

if mean_diff > mean_same * 5:
    print(
        "PASS: inter-video variation is much larger than phase variation."
    )
else:
    print(
        "WARNING: inter-video separation is weak."
    )

if feature_std > 0.01:
    print(
        "PASS: World representation has non-trivial feature variance."
    )
else:
    print(
        "WARNING: possible representation collapse."
    )

print()
print("DONE")