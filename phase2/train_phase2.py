import random
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from factorizer import WorldGridFactorizer

from temporal_predictor import (
    TemporalPredictor,
    multi_horizon_loss,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

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


OUTPUT_DIR = (
    ROOT
    / "checkpoints"
    / "phase2"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA
# ============================================================

NUM_TRAIN_CLIPS = 256

VAL_START = 72
NUM_VAL_CLIPS = 32


# ============================================================
# VIDEO / V-JEPA
# ============================================================

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


# ============================================================
# TEMPORAL TASK
#
# context:
# t0 t1 t2 t3
#
# predict:
# t4 t5 t6
# ============================================================

CONTEXT_LENGTH = 4

TARGET_INDICES = [
    4,
    5,
    6,
]

NUM_HORIZONS = len(
    TARGET_INDICES
)


# ============================================================
# PHASE INTERVENTIONS
# ============================================================

TRAIN_PHASES = [
    0,
    4,
    8,
    12,
]


# ============================================================
# PREDICTOR
# ============================================================

PREDICTOR_HIDDEN_DIM = 384
PREDICTOR_LAYERS = 3
PREDICTOR_HEADS = 6

EPOCHS = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0


# ============================================================
# IMAGENET NORMALIZATION
# ============================================================

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
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# LOAD PHASE-1 FACTORIZER
# ============================================================

print()
print("=" * 80)
print("Loading frozen Phase-1 factorizer")
print("=" * 80)

phase1_ckpt = torch.load(
    PHASE1_CHECKPOINT,
    map_location=DEVICE,
)

phase1_cfg = phase1_ckpt[
    "config"
]

WORLD_DIM = phase1_cfg[
    "world_dim"
]

GRID_DIM = phase1_cfg[
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


print("World dim:", WORLD_DIM)
print("Grid dim:", GRID_DIM)
print("Factorizer frozen.")


# ============================================================
# BUILD PREDICTORS
# ============================================================

raw_predictor = TemporalPredictor(
    input_dim=768,
    output_dim=768,
    hidden_dim=PREDICTOR_HIDDEN_DIM,
    num_layers=PREDICTOR_LAYERS,
    num_heads=PREDICTOR_HEADS,
    num_horizons=NUM_HORIZONS,
).to(DEVICE)


world_predictor = TemporalPredictor(
    input_dim=WORLD_DIM,
    output_dim=WORLD_DIM,
    hidden_dim=PREDICTOR_HIDDEN_DIM,
    num_layers=PREDICTOR_LAYERS,
    num_heads=PREDICTOR_HEADS,
    num_horizons=NUM_HORIZONS,
).to(DEVICE)


def count_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


print()
print(
    "Raw predictor trainable params:",
    f"{count_params(raw_predictor)/1e6:.3f}M",
)

print(
    "World predictor trainable params:",
    f"{count_params(world_predictor)/1e6:.3f}M",
)


# ============================================================
# OPTIMIZERS
# ============================================================

raw_optimizer = AdamW(
    raw_predictor.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


world_optimizer = AdamW(
    world_predictor.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 80)
print("Loading local MOVi-C")
print("=" * 80)

builder = tfds.builder(
    "movi_c/256x256",
    data_dir=str(DATA_DIR),
)


train_ds = builder.as_dataset(
    split=f"train[:{NUM_TRAIN_CLIPS}]",
    shuffle_files=False,
)


val_ds = builder.as_dataset(
    split=(
        f"validation["
        f"{VAL_START}:"
        f"{VAL_START + NUM_VAL_CLIPS}]"
    ),
    shuffle_files=False,
)


print(
    "Dataset:",
    builder.info.full_name,
)

print(
    "Train clips:",
    NUM_TRAIN_CLIPS,
)

print(
    "Validation clips:",
    NUM_VAL_CLIPS,
)


# ============================================================
# VIDEO PREPROCESSING
# ============================================================

def prepare_video(sample):

    video = (
        sample["video"]
        .numpy()
        [:NUM_FRAMES]
    )

    # [T,H,W,C]
    video = torch.from_numpy(video)

    # [C,T,H,W]
    video = video.permute(
        3,
        0,
        1,
        2,
    )

    # [1,C,T,H,W]
    video = video.unsqueeze(0)

    video = (
        video.float()
        / 255.0
    )

    B, C, T, H, W = (
        video.shape
    )

    # resize every frame
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

    video = x.permute(
        0,
        2,
        1,
        3,
        4,
    )

    mean = IMAGENET_MEAN.to(
        video.device,
        video.dtype,
    )

    std = IMAGENET_STD.to(
        video.device,
        video.dtype,
    )

    video = (
        video - mean
    ) / std

    return video


# ============================================================
# SHIFT VIDEO
# ============================================================

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


# ============================================================
# ENCODE V-JEPA
# ============================================================

def encode_video(video):

    with torch.no_grad():
        z = encoder(video)

    z = z.detach()

    B, N, D = z.shape

    expected_tokens = (
        TEMPORAL_TOKENS
        * SPATIAL_TOKENS
        * SPATIAL_TOKENS
    )

    if N != expected_tokens:
        raise RuntimeError(
            f"Unexpected token count: "
            f"{N}, expected "
            f"{expected_tokens}"
        )

    z = z.reshape(
        B,
        TEMPORAL_TOKENS,
        SPATIAL_TOKENS,
        SPATIAL_TOKENS,
        D,
    )

    return z


# ============================================================
# LATENT ALIGNMENT
#
# Undo spatial translation at latent level
# so different phase interventions correspond
# to approximately the same world coordinates.
# ============================================================

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

    x = x.permute(
        0,
        1,
        3,
        4,
        2,
    )

    return x


# ============================================================
# REPRESENTATION EXTRACTION
# ============================================================

def extract_representations(
    video,
    phase,
):

    shifted_video = shift_clip(
        video,
        phase,
    )

    z = encode_video(
        shifted_video
    )

    z = latent_resample(
        z,
        phase,
    )

    with torch.no_grad():
        world, grid, logits = (
            factorizer(z)
        )

    return (
        z.detach(),
        world.detach(),
    )


# ============================================================
# SPLIT TEMPORAL TOKENS
# ============================================================

def split_temporal(
    representation,
):

    context = representation[
        :,
        :CONTEXT_LENGTH,
    ]

    targets = []

    for idx in TARGET_INDICES:

        targets.append(
            representation[
                :,
                idx,
            ]
        )

    return (
        context,
        targets,
    )


# ============================================================
# COPY-LAST BASELINE
# ============================================================

def copy_last_loss(
    context,
    targets,
):

    last_context = context[
        :,
        -1,
    ]

    predictions = [
        last_context
        for _ in targets
    ]

    return multi_horizon_loss(
        predictions,
        targets,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate():

    raw_predictor.eval()
    world_predictor.eval()

    phases = [
        0,
        8,
    ]

    results = {}

    for phase in phases:

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

    for sample in val_ds:

        video = (
            prepare_video(sample)
            .to(DEVICE)
        )

        for phase in phases:

            z, world = (
                extract_representations(
                    video,
                    phase,
                )
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

                raw_h_loss = (
                    multi_horizon_loss(
                        [
                            raw_predictions[h]
                        ],
                        [
                            raw_targets[h]
                        ],
                    )[
                        "total"
                    ].item()
                )

                world_h_loss = (
                    multi_horizon_loss(
                        [
                            world_predictions[h]
                        ],
                        [
                            world_targets[h]
                        ],
                    )[
                        "total"
                    ].item()
                )

                r[
                    "raw_horizons"
                ][h].append(
                    raw_h_loss
                )

                r[
                    "world_horizons"
                ][h].append(
                    world_h_loss
                )

    summary = {}

    for phase in phases:

        r = results[phase]

        raw_pred = float(
            np.mean(
                r[
                    "raw_pred"
                ]
            )
        )

        raw_copy = float(
            np.mean(
                r[
                    "raw_copy"
                ]
            )
        )

        world_pred = float(
            np.mean(
                r[
                    "world_pred"
                ]
            )
        )

        world_copy = float(
            np.mean(
                r[
                    "world_copy"
                ]
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

            "raw_horizons":
                [
                    float(
                        np.mean(x)
                    )
                    for x
                    in r[
                        "raw_horizons"
                    ]
                ],

            "world_horizons":
                [
                    float(
                        np.mean(x)
                    )
                    for x
                    in r[
                        "world_horizons"
                    ]
                ],
        }

    raw_predictor.train()
    world_predictor.train()

    return summary


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 80)
print("Starting Phase-2 training")
print("=" * 80)

best_score = float(
    "inf"
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    raw_predictor.train()
    world_predictor.train()

    raw_epoch_losses = []
    world_epoch_losses = []

    print()
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    for clip_idx, sample in enumerate(
        train_ds
    ):

        video = (
            prepare_video(sample)
            .to(DEVICE)
        )

        phase = random.choice(
            TRAIN_PHASES
        )

        z, world = (
            extract_representations(
                video,
                phase,
            )
        )

        (
            raw_context,
            raw_targets,
        ) = split_temporal(z)

        (
            world_context,
            world_targets,
        ) = split_temporal(world)

        # ====================================================
        # RAW
        # ====================================================

        raw_optimizer.zero_grad(
            set_to_none=True
        )

        raw_predictions = (
            raw_predictor(
                raw_context
            )
        )

        raw_losses = (
            multi_horizon_loss(
                raw_predictions,
                raw_targets,
            )
        )

        raw_losses[
            "total"
        ].backward()

        torch.nn.utils.clip_grad_norm_(
            raw_predictor.parameters(),
            GRAD_CLIP,
        )

        raw_optimizer.step()

        # ====================================================
        # WORLD
        # ====================================================

        world_optimizer.zero_grad(
            set_to_none=True
        )

        world_predictions = (
            world_predictor(
                world_context
            )
        )

        world_losses = (
            multi_horizon_loss(
                world_predictions,
                world_targets,
            )
        )

        world_losses[
            "total"
        ].backward()

        torch.nn.utils.clip_grad_norm_(
            world_predictor.parameters(),
            GRAD_CLIP,
        )

        world_optimizer.step()

        raw_epoch_losses.append(
            raw_losses[
                "total"
            ].item()
        )

        world_epoch_losses.append(
            world_losses[
                "total"
            ].item()
        )

        if (
            clip_idx % 16 == 0
        ):

            print(
                f"clip={clip_idx:03d}"
                f" | phase={phase:02d}"
                f" | raw={raw_losses['total'].item():.6f}"
                f" | world={world_losses['total'].item():.6f}"
            )

    print()

    print(
        "Train raw loss:",
        f"{np.mean(raw_epoch_losses):.6f}"
    )

    print(
        "Train world loss:",
        f"{np.mean(world_epoch_losses):.6f}"
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    val = evaluate()

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    for phase in [
        0,
        8,
    ]:

        r = val[phase]

        print()
        print(
            f"Phase = {phase}px"
        )

        print(
            f"Raw predictor       : "
            f"{r['raw_pred']:.6f}"
        )

        print(
            f"Raw copy baseline   : "
            f"{r['raw_copy']:.6f}"
        )

        print(
            f"Raw relative gain   : "
            f"{r['raw_gain'] * 100:.2f}%"
        )

        print()

        print(
            f"World predictor     : "
            f"{r['world_pred']:.6f}"
        )

        print(
            f"World copy baseline : "
            f"{r['world_copy']:.6f}"
        )

        print(
            f"World relative gain : "
            f"{r['world_gain'] * 100:.2f}%"
        )

        print()

        print(
            "Raw horizons:",
            [
                round(x, 6)
                for x in r[
                    "raw_horizons"
                ]
            ],
        )

        print(
            "World horizons:",
            [
                round(x, 6)
                for x in r[
                    "world_horizons"
                ]
            ],
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = {
        "epoch":
            epoch,

        "raw_predictor":
            raw_predictor.state_dict(),

        "world_predictor":
            world_predictor.state_dict(),

        "validation":
            val,

        "config": {
            "context_length":
                CONTEXT_LENGTH,

            "target_indices":
                TARGET_INDICES,

            "hidden_dim":
                PREDICTOR_HIDDEN_DIM,

            "layers":
                PREDICTOR_LAYERS,

            "heads":
                PREDICTOR_HEADS,

            "world_dim":
                WORLD_DIM,

            "grid_dim":
                GRID_DIM,
        },
    }

    torch.save(
        checkpoint,
        OUTPUT_DIR
        / "last.pt",
    )

    # choose best based on normalized gains,
    # not raw cross-space losses
    score = -(
        val[0][
            "raw_gain"
        ]
        +
        val[0][
            "world_gain"
        ]
    )

    if score < best_score:

        best_score = score

        torch.save(
            checkpoint,
            OUTPUT_DIR
            / "best.pt",
        )

        print()
        print(
            "Saved new best checkpoint."
        )


print()
print("=" * 80)
print("PHASE-2 TRAINING FINISHED")
print("=" * 80)