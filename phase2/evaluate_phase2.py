from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

import torch
import torch.nn.functional as F

from factorizer import WorldGridFactorizer

from temporal_predictor import (
    TemporalPredictor,
    prediction_loss,
    multi_horizon_loss,
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


ROOT = Path(
    "/data/mi2-interns/ruiyu/R-JEPA"
)


VJEPA_REPO = (
    ROOT / "vjepa2"
)


DATA_DIR = (
    ROOT / "data/tfds"
)


PHASE1_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "phase1"
    / "best.pt"
)


PHASE2_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "phase2"
    / "best.pt"
)


NUM_FRAMES = 16
IMAGE_SIZE = 384
PATCH_SIZE = 16
TUBELET_SIZE = 2

TEMPORAL_TOKENS = (
    NUM_FRAMES
    // TUBELET_SIZE
)

SPATIAL_TOKENS = (
    IMAGE_SIZE
    // PATCH_SIZE
)


CONTEXT_LENGTH = 4

TARGET_INDICES = [
    4,
    5,
    6,
]

NUM_HORIZONS = len(
    TARGET_INDICES
)


EVAL_START = 104
NUM_EVAL_CLIPS = 64


TEST_PHASES = [
    0,
    4,
    8,
    12,
]


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(
    1, 3, 1, 1, 1
)


IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(
    1, 3, 1, 1, 1
)


# ============================================================
# LOAD V-JEPA
# ============================================================

print("=" * 80)
print("Loading frozen V-JEPA")
print("=" * 80)

encoder, unused_predictor = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)

encoder = encoder.to(DEVICE)
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

del unused_predictor


# ============================================================
# LOAD PHASE 1
# ============================================================

phase1_ckpt = torch.load(
    PHASE1_CHECKPOINT,
    map_location=DEVICE,
)

cfg1 = phase1_ckpt[
    "config"
]

WORLD_DIM = cfg1[
    "world_dim"
]

GRID_DIM = cfg1[
    "grid_dim"
]


factorizer = WorldGridFactorizer(
    input_dim=768,
    hidden_dim=768,
    world_dim=WORLD_DIM,
    grid_dim=GRID_DIM,
    num_grid_classes=4,
).to(DEVICE)


factorizer.load_state_dict(
    phase1_ckpt[
        "factorizer"
    ]
)

factorizer.eval()

for p in factorizer.parameters():
    p.requires_grad = False


# ============================================================
# LOAD PHASE 2
# ============================================================

phase2_ckpt = torch.load(
    PHASE2_CHECKPOINT,
    map_location=DEVICE,
)

cfg2 = phase2_ckpt[
    "config"
]


raw_predictor = TemporalPredictor(
    input_dim=768,
    output_dim=768,
    hidden_dim=cfg2[
        "hidden_dim"
    ],
    num_layers=cfg2[
        "layers"
    ],
    num_heads=cfg2[
        "heads"
    ],
    num_horizons=NUM_HORIZONS,
).to(DEVICE)


world_predictor = TemporalPredictor(
    input_dim=WORLD_DIM,
    output_dim=WORLD_DIM,
    hidden_dim=cfg2[
        "hidden_dim"
    ],
    num_layers=cfg2[
        "layers"
    ],
    num_heads=cfg2[
        "heads"
    ],
    num_horizons=NUM_HORIZONS,
).to(DEVICE)


raw_predictor.load_state_dict(
    phase2_ckpt[
        "raw_predictor"
    ]
)

world_predictor.load_state_dict(
    phase2_ckpt[
        "world_predictor"
    ]
)

raw_predictor.eval()
world_predictor.eval()


# ============================================================
# DATASET
# ============================================================

builder = tfds.builder(
    "movi_c/256x256",
    data_dir=str(DATA_DIR),
)


dataset = builder.as_dataset(
    split=(
        f"validation["
        f"{EVAL_START}:"
        f"{EVAL_START + NUM_EVAL_CLIPS}]"
    ),
    shuffle_files=False,
)


print(
    "Evaluation clips:",
    NUM_EVAL_CLIPS,
)


# ============================================================
# HELPERS
# ============================================================

def prepare_video(sample):

    video = (
        sample["video"]
        .numpy()
        [:NUM_FRAMES]
    )

    video = torch.from_numpy(
        video
    )

    video = video.permute(
        3,
        0,
        1,
        2,
    )

    video = video.unsqueeze(0)

    video = (
        video.float()
        / 255.0
    )

    B, C, T, H, W = (
        video.shape
    )

    x = video.permute(
        0,
        2,
        1,
        3,
        4,
    )

    x = x.reshape(
        B * T,
        C,
        H,
        W,
    )

    x = F.interpolate(
        x,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    x = x.reshape(
        B,
        T,
        C,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    x = x.permute(
        0,
        2,
        1,
        3,
        4,
    )

    mean = IMAGENET_MEAN.to(
        x.device,
        x.dtype,
    )

    std = IMAGENET_STD.to(
        x.device,
        x.dtype,
    )

    return (
        x - mean
    ) / std


def shift_clip(
    video,
    dx,
):

    if dx == 0:
        return video

    B, C, T, H, W = (
        video.shape
    )

    x = video.permute(
        0,
        2,
        1,
        3,
        4,
    )

    x = x.reshape(
        B * T,
        C,
        H,
        W,
    )

    theta = torch.tensor(
        [
            [
                1.0,
                0.0,
                -2.0 * dx / (W - 1),
            ],
            [
                0.0,
                1.0,
                0.0,
            ],
        ],
        dtype=x.dtype,
        device=x.device,
    )

    theta = theta.unsqueeze(0).repeat(
        B * T,
        1,
        1,
    )

    grid = F.affine_grid(
        theta,
        x.size(),
        align_corners=True,
    )

    x = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )

    x = x.reshape(
        B,
        T,
        C,
        H,
        W,
    )

    return x.permute(
        0,
        2,
        1,
        3,
        4,
    )


def encode_video(video):

    with torch.no_grad():
        z = encoder(video)

    B, N, D = z.shape

    z = z.reshape(
        B,
        TEMPORAL_TOKENS,
        SPATIAL_TOKENS,
        SPATIAL_TOKENS,
        D,
    )

    return z


def latent_resample(
    latent,
    dx_pixels,
):

    if dx_pixels == 0:
        return latent

    B, T, H, W, D = (
        latent.shape
    )

    x = latent.permute(
        0,
        1,
        4,
        2,
        3,
    )

    x = x.reshape(
        B * T,
        D,
        H,
        W,
    )

    dx_tokens = (
        dx_pixels
        / PATCH_SIZE
    )

    normalized_shift = (
        2.0
        * dx_tokens
        / (W - 1)
    )

    ys, xs = torch.meshgrid(
        torch.linspace(
            -1.0,
            1.0,
            H,
            device=x.device,
            dtype=x.dtype,
        ),
        torch.linspace(
            -1.0,
            1.0,
            W,
            device=x.device,
            dtype=x.dtype,
        ),
        indexing="ij",
    )

    grid = torch.stack(
        [
            xs + normalized_shift,
            ys,
        ],
        dim=-1,
    )

    grid = grid.unsqueeze(0).repeat(
        B * T,
        1,
        1,
        1,
    )

    x = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    x = x.reshape(
        B,
        T,
        D,
        H,
        W,
    )

    return x.permute(
        0,
        1,
        3,
        4,
        2,
    )


def split_temporal(x):

    context = x[
        :,
        :CONTEXT_LENGTH,
    ]

    targets = [
        x[:, idx]
        for idx in TARGET_INDICES
    ]

    return (
        context,
        targets,
    )


def copy_last_loss(
    context,
    targets,
):

    last = context[
        :,
        -1,
    ]

    predictions = [
        last
        for _ in targets
    ]

    return multi_horizon_loss(
        predictions,
        targets,
    )


# ============================================================
# EVALUATE
# ============================================================

results = {}

for phase in TEST_PHASES:

    results[phase] = {
        "raw_pred": [],
        "raw_copy": [],
        "world_pred": [],
        "world_copy": [],

        "raw_horizons": [
            [],
            [],
            [],
        ],

        "world_horizons": [
            [],
            [],
            [],
        ],
    }


print()
print("=" * 80)
print("Starting Phase-2 evaluation")
print("=" * 80)


with torch.no_grad():

    for clip_idx, sample in enumerate(
        dataset
    ):

        video = (
            prepare_video(sample)
            .to(DEVICE)
        )

        for phase in TEST_PHASES:

            shifted = shift_clip(
                video,
                phase,
            )

            z = encode_video(
                shifted
            )

            z = latent_resample(
                z,
                phase,
            )

            world, _, _ = (
                factorizer(z)
            )

            (
                raw_context,
                raw_targets,
            ) = split_temporal(z)

            (
                world_context,
                world_targets,
            ) = split_temporal(world)

            raw_predictions = (
                raw_predictor(
                    raw_context
                )
            )

            world_predictions = (
                world_predictor(
                    world_context
                )
            )

            raw_loss = (
                multi_horizon_loss(
                    raw_predictions,
                    raw_targets,
                )
            )

            world_loss = (
                multi_horizon_loss(
                    world_predictions,
                    world_targets,
                )
            )

            raw_copy = (
                copy_last_loss(
                    raw_context,
                    raw_targets,
                )
            )

            world_copy = (
                copy_last_loss(
                    world_context,
                    world_targets,
                )
            )

            r = results[phase]

            r["raw_pred"].append(
                raw_loss[
                    "total"
                ].item()
            )

            r["world_pred"].append(
                world_loss[
                    "total"
                ].item()
            )

            r["raw_copy"].append(
                raw_copy[
                    "total"
                ].item()
            )

            r["world_copy"].append(
                world_copy[
                    "total"
                ].item()
            )

            for h in range(
                NUM_HORIZONS
            ):

                raw_h = prediction_loss(
                    raw_predictions[h],
                    raw_targets[h],
                )[
                    "total"
                ].item()

                world_h = prediction_loss(
                    world_predictions[h],
                    world_targets[h],
                )[
                    "total"
                ].item()

                r[
                    "raw_horizons"
                ][h].append(
                    raw_h
                )

                r[
                    "world_horizons"
                ][h].append(
                    world_h
                )

        if clip_idx % 8 == 0:
            print(
                f"Finished clip {clip_idx}"
            )


# ============================================================
# SUMMARY
# ============================================================

summary = {}

print()
print("=" * 100)
print("PHASE-2 RESULTS")
print("=" * 100)


for phase in TEST_PHASES:

    r = results[phase]

    raw_pred = float(
        np.mean(
            r["raw_pred"]
        )
    )

    raw_copy = float(
        np.mean(
            r["raw_copy"]
        )
    )

    world_pred = float(
        np.mean(
            r["world_pred"]
        )
    )

    world_copy = float(
        np.mean(
            r["world_copy"]
        )
    )

    raw_gain = (
        raw_copy
        - raw_pred
    ) / max(
        raw_copy,
        1e-8,
    )

    world_gain = (
        world_copy
        - world_pred
    ) / max(
        world_copy,
        1e-8,
    )

    summary[phase] = {
        "raw_pred":
            raw_pred,

        "raw_copy":
            raw_copy,

        "raw_gain":
            raw_gain,

        "world_pred":
            world_pred,

        "world_copy":
            world_copy,

        "world_gain":
            world_gain,
    }

    print()
    print(
        f"PHASE = {phase}px"
    )

    print(
        "-" * 60
    )

    print(
        f"Raw predictor        : "
        f"{raw_pred:.6f}"
    )

    print(
        f"Raw copy baseline    : "
        f"{raw_copy:.6f}"
    )

    print(
        f"Raw relative gain    : "
        f"{raw_gain * 100:.2f}%"
    )

    print()

    print(
        f"World predictor      : "
        f"{world_pred:.6f}"
    )

    print(
        f"World copy baseline  : "
        f"{world_copy:.6f}"
    )

    print(
        f"World relative gain  : "
        f"{world_gain * 100:.2f}%"
    )

    print()

    print(
        "Raw horizon losses:"
    )

    for h in range(
        NUM_HORIZONS
    ):

        print(
            f"  t+{h+1}: "
            f"{np.mean(r['raw_horizons'][h]):.6f}"
        )

    print(
        "World horizon losses:"
    )

    for h in range(
        NUM_HORIZONS
    ):

        print(
            f"  t+{h+1}: "
            f"{np.mean(r['world_horizons'][h]):.6f}"
        )


# ============================================================
# PHASE ROBUSTNESS
# ============================================================

raw_phase_ratio = (
    summary[8][
        "raw_pred"
    ]
    /
    summary[0][
        "raw_pred"
    ]
)


world_phase_ratio = (
    summary[8][
        "world_pred"
    ]
    /
    summary[0][
        "world_pred"
    ]
)


print()
print("=" * 100)
print("PATCH-PHASE ROBUSTNESS")
print("=" * 100)

print(
    f"Raw 8px / 0px loss ratio   : "
    f"{raw_phase_ratio:.4f}x"
)

print(
    f"World 8px / 0px loss ratio : "
    f"{world_phase_ratio:.4f}x"
)


print()
print("=" * 100)
print("MAIN COMPARISON")
print("=" * 100)

print(
    f"Raw relative gain @0px   : "
    f"{summary[0]['raw_gain'] * 100:.2f}%"
)

print(
    f"World relative gain @0px : "
    f"{summary[0]['world_gain'] * 100:.2f}%"
)

print()

print(
    f"Raw relative gain @8px   : "
    f"{summary[8]['raw_gain'] * 100:.2f}%"
)

print(
    f"World relative gain @8px : "
    f"{summary[8]['world_gain'] * 100:.2f}%"
)

print()

print(
    f"Raw phase degradation   : "
    f"{raw_phase_ratio:.4f}x"
)

print(
    f"World phase degradation : "
    f"{world_phase_ratio:.4f}x"
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)