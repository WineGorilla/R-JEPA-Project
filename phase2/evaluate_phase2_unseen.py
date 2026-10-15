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


# ============================================================
# VIDEO / TOKENIZATION
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
# TEMPORAL PREDICTION TASK
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
# EVALUATION DATA
#
# Phase-2 validation:
# validation[72:104]
#
# This evaluation:
# validation[104:168]
#
# Therefore these clips were not used for
# Phase-2 model selection.
# ============================================================

EVAL_START = 104

NUM_EVAL_CLIPS = 64


# ============================================================
# PHASES
#
# Seen during Phase-2 training:
# 0, 4, 8, 12
#
# Unseen:
# 2, 6, 10, 14, 16
# ============================================================

TRAINED_PHASES = {
    0,
    4,
    8,
    12,
}


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


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = torch.tensor(
    [
        0.485,
        0.456,
        0.406,
    ]
).view(
    1,
    3,
    1,
    1,
    1,
)


IMAGENET_STD = torch.tensor(
    [
        0.229,
        0.224,
        0.225,
    ]
).view(
    1,
    3,
    1,
    1,
    1,
)


# ============================================================
# LOAD V-JEPA
# ============================================================

print("=" * 100)
print("Loading frozen V-JEPA")
print("=" * 100)


encoder, unused_predictor = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)


encoder = encoder.to(
    DEVICE
)


encoder.eval()


for p in encoder.parameters():
    p.requires_grad = False


del unused_predictor


print(
    "Device:",
    DEVICE,
)


if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# LOAD PHASE-1 FACTORIZER
# ============================================================

print()

print("=" * 100)
print("Loading frozen Phase-1 factorizer")
print("=" * 100)


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
).to(
    DEVICE
)


factorizer.load_state_dict(
    phase1_ckpt[
        "factorizer"
    ]
)


factorizer.eval()


for p in factorizer.parameters():
    p.requires_grad = False


print(
    "World dim:",
    WORLD_DIM,
)


print(
    "Grid dim:",
    GRID_DIM,
)


# ============================================================
# LOAD PHASE-2 PREDICTORS
# ============================================================

print()

print("=" * 100)
print("Loading Phase-2 temporal predictors")
print("=" * 100)


phase2_ckpt = torch.load(
    PHASE2_CHECKPOINT,
    map_location=DEVICE,
)


phase2_cfg = phase2_ckpt[
    "config"
]


raw_predictor = TemporalPredictor(
    input_dim=768,
    output_dim=768,
    hidden_dim=phase2_cfg[
        "hidden_dim"
    ],
    num_layers=phase2_cfg[
        "layers"
    ],
    num_heads=phase2_cfg[
        "heads"
    ],
    num_horizons=NUM_HORIZONS,
).to(
    DEVICE
)


world_predictor = TemporalPredictor(
    input_dim=WORLD_DIM,
    output_dim=WORLD_DIM,
    hidden_dim=phase2_cfg[
        "hidden_dim"
    ],
    num_layers=phase2_cfg[
        "layers"
    ],
    num_heads=phase2_cfg[
        "heads"
    ],
    num_horizons=NUM_HORIZONS,
).to(
    DEVICE
)


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


print(
    "Phase-2 checkpoint epoch:",
    phase2_ckpt.get(
        "epoch",
        "unknown",
    ),
)


# ============================================================
# DATASET
# ============================================================

print()

print("=" * 100)
print("Loading local MOVi-C")
print("=" * 100)


builder = tfds.builder(
    "movi_c/256x256",
    data_dir=str(
        DATA_DIR
    ),
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
    "Dataset:",
    builder.info.full_name,
)


print(
    "Evaluation clips:",
    NUM_EVAL_CLIPS,
)


print(
    "Test phases:",
    TEST_PHASES,
)


# ============================================================
# PREPROCESS VIDEO
# ============================================================

def prepare_video(
    sample,
):

    video = (
        sample["video"]
        .numpy()
        [:NUM_FRAMES]
    )


    # [T,H,W,C]
    video = torch.from_numpy(
        video
    )


    # [C,T,H,W]
    video = video.permute(
        3,
        0,
        1,
        2,
    )


    # [1,C,T,H,W]
    video = video.unsqueeze(
        0
    )


    video = (
        video.float()
        / 255.0
    )


    B, C, T, H, W = (
        video.shape
    )


    # --------------------------------------------------------
    # Resize frames:
    #
    # 256 x 256
    # ->
    # 384 x 384
    # --------------------------------------------------------

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
        video
        - mean
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
                -2.0
                * dx
                / (W - 1),
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


    theta = theta.unsqueeze(
        0
    ).repeat(
        B * T,
        1,
        1,
    )


    grid = F.affine_grid(
        theta,
        x.size(),
        align_corners=True,
    )


    shifted = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=True,
    )


    shifted = shifted.reshape(
        B,
        T,
        C,
        H,
        W,
    )


    shifted = shifted.permute(
        0,
        2,
        1,
        3,
        4,
    )


    return shifted


# ============================================================
# ENCODE VIDEO
# ============================================================

def encode_video(
    video,
):

    with torch.no_grad():

        z = encoder(
            video
        )


    B, N, D = (
        z.shape
    )


    expected_tokens = (
        TEMPORAL_TOKENS
        * SPATIAL_TOKENS
        * SPATIAL_TOKENS
    )


    if N != expected_tokens:

        raise RuntimeError(
            f"Unexpected token count "
            f"{N}; expected "
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
# LATENT RESAMPLING / ALIGNMENT
#
# Align shifted latent representations back
# toward common world coordinates.
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
            xs
            + normalized_shift,
            ys,
        ],
        dim=-1,
    )


    grid = grid.unsqueeze(
        0
    ).repeat(
        B * T,
        1,
        1,
        1,
    )


    aligned = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


    aligned = aligned.reshape(
        B,
        T,
        D,
        H,
        W,
    )


    aligned = aligned.permute(
        0,
        1,
        3,
        4,
        2,
    )


    return aligned


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def split_temporal(
    representation,
):

    context = representation[
        :,
        :CONTEXT_LENGTH,
    ]


    targets = [
        representation[
            :,
            idx,
        ]
        for idx
        in TARGET_INDICES
    ]


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
# RESULTS STORAGE
# ============================================================

results = {}


for phase in TEST_PHASES:

    results[
        phase
    ] = {

        "raw_pred": [],

        "raw_copy": [],

        "world_pred": [],

        "world_copy": [],


        "raw_cosine": [],

        "world_cosine": [],


        "raw_l1": [],

        "world_l1": [],


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


        "raw_copy_horizons": [
            [],
            [],
            [],
        ],


        "world_copy_horizons": [
            [],
            [],
            [],
        ],
    }


# ============================================================
# EVALUATION
# ============================================================

print()

print("=" * 100)
print("Starting unseen-phase evaluation")
print("=" * 100)


with torch.no_grad():

    for clip_idx, sample in enumerate(
        dataset
    ):

        video = (
            prepare_video(
                sample
            )
            .to(
                DEVICE
            )
        )


        for phase in TEST_PHASES:

            # ------------------------------------------------
            # Apply tokenizer intervention
            # ------------------------------------------------

            shifted_video = (
                shift_clip(
                    video,
                    phase,
                )
            )


            # ------------------------------------------------
            # Frozen V-JEPA
            # ------------------------------------------------

            z = encode_video(
                shifted_video
            )


            # ------------------------------------------------
            # Align representation
            # ------------------------------------------------

            z = latent_resample(
                z,
                phase,
            )


            # ------------------------------------------------
            # Frozen World/Grid Factorizer
            # ------------------------------------------------

            world, _, _ = (
                factorizer(
                    z
                )
            )


            # ------------------------------------------------
            # Split past / future
            # ------------------------------------------------

            (
                raw_context,
                raw_targets,
            ) = split_temporal(
                z
            )


            (
                world_context,
                world_targets,
            ) = split_temporal(
                world
            )


            # ------------------------------------------------
            # Predict future
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Aggregate predictor losses
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Copy-last baselines
            # ------------------------------------------------

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


            r = results[
                phase
            ]


            r[
                "raw_pred"
            ].append(
                raw_loss[
                    "total"
                ].item()
            )


            r[
                "world_pred"
            ].append(
                world_loss[
                    "total"
                ].item()
            )


            r[
                "raw_copy"
            ].append(
                raw_copy[
                    "total"
                ].item()
            )


            r[
                "world_copy"
            ].append(
                world_copy[
                    "total"
                ].item()
            )


            r[
                "raw_cosine"
            ].append(
                raw_loss[
                    "cosine"
                ].item()
            )


            r[
                "world_cosine"
            ].append(
                world_loss[
                    "cosine"
                ].item()
            )


            r[
                "raw_l1"
            ].append(
                raw_loss[
                    "smooth_l1"
                ].item()
            )


            r[
                "world_l1"
            ].append(
                world_loss[
                    "smooth_l1"
                ].item()
            )


            # ------------------------------------------------
            # Per-horizon statistics
            # ------------------------------------------------

            raw_last = (
                raw_context[
                    :,
                    -1,
                ]
            )


            world_last = (
                world_context[
                    :,
                    -1,
                ]
            )


            for h in range(
                NUM_HORIZONS
            ):

                raw_h_loss = (
                    prediction_loss(
                        raw_predictions[h],
                        raw_targets[h],
                    )[
                        "total"
                    ].item()
                )


                world_h_loss = (
                    prediction_loss(
                        world_predictions[h],
                        world_targets[h],
                    )[
                        "total"
                    ].item()
                )


                raw_copy_h_loss = (
                    prediction_loss(
                        raw_last,
                        raw_targets[h],
                    )[
                        "total"
                    ].item()
                )


                world_copy_h_loss = (
                    prediction_loss(
                        world_last,
                        world_targets[h],
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


                r[
                    "raw_copy_horizons"
                ][h].append(
                    raw_copy_h_loss
                )


                r[
                    "world_copy_horizons"
                ][h].append(
                    world_copy_h_loss
                )


        if (
            clip_idx % 8
            == 0
        ):

            print(
                f"Finished clip "
                f"{clip_idx}"
            )


# ============================================================
# AGGREGATE
# ============================================================

summary = {}


print()

print("=" * 120)
print("PHASE-2B UNSEEN PHASE RESULTS")
print("=" * 120)


for phase in TEST_PHASES:

    r = results[
        phase
    ]


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


    advantage_pp = (
        world_gain
        - raw_gain
    ) * 100.0


    summary[
        phase
    ] = {

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

        "advantage_pp":
            advantage_pp,

        "raw_cosine":
            float(
                np.mean(
                    r[
                        "raw_cosine"
                    ]
                )
            ),

        "world_cosine":
            float(
                np.mean(
                    r[
                        "world_cosine"
                    ]
                )
            ),

        "raw_l1":
            float(
                np.mean(
                    r[
                        "raw_l1"
                    ]
                )
            ),

        "world_l1":
            float(
                np.mean(
                    r[
                        "world_l1"
                    ]
                )
            ),
    }


    status = (
        "SEEN"
        if phase
        in TRAINED_PHASES
        else "UNSEEN"
    )


    print()

    print(
        f"PHASE = "
        f"{phase:02d}px "
        f"[{status}]"
    )


    print(
        "-" * 80
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
        f"World advantage      : "
        f"{advantage_pp:+.2f} pp"
    )


    print()


    print(
        "Raw horizon predictor / copy:"
    )


    for h in range(
        NUM_HORIZONS
    ):

        pred_h = float(
            np.mean(
                r[
                    "raw_horizons"
                ][h]
            )
        )


        copy_h = float(
            np.mean(
                r[
                    "raw_copy_horizons"
                ][h]
            )
        )


        gain_h = (
            copy_h
            - pred_h
        ) / max(
            copy_h,
            1e-8,
        )


        print(
            f"  t+{h + 1}: "
            f"pred={pred_h:.6f}"
            f" | copy={copy_h:.6f}"
            f" | gain="
            f"{gain_h * 100:.2f}%"
        )


    print(
        "World horizon predictor / copy:"
    )


    for h in range(
        NUM_HORIZONS
    ):

        pred_h = float(
            np.mean(
                r[
                    "world_horizons"
                ][h]
            )
        )


        copy_h = float(
            np.mean(
                r[
                    "world_copy_horizons"
                ][h]
            )
        )


        gain_h = (
            copy_h
            - pred_h
        ) / max(
            copy_h,
            1e-8,
        )


        print(
            f"  t+{h + 1}: "
            f"pred={pred_h:.6f}"
            f" | copy={copy_h:.6f}"
            f" | gain="
            f"{gain_h * 100:.2f}%"
        )


# ============================================================
# COMPACT TABLE
# ============================================================

print()

print("=" * 120)
print("COMPACT SUMMARY")
print("=" * 120)


print(
    f"{'Phase':>6} | "
    f"{'Type':>6} | "
    f"{'Raw Gain':>10} | "
    f"{'World Gain':>11} | "
    f"{'World Adv.':>11}"
)


print(
    "-" * 65
)


for phase in TEST_PHASES:

    s = summary[
        phase
    ]


    phase_type = (
        "seen"
        if phase
        in TRAINED_PHASES
        else "unseen"
    )


    print(
        f"{phase:>4}px | "
        f"{phase_type:>6} | "
        f"{s['raw_gain'] * 100:>9.2f}% | "
        f"{s['world_gain'] * 100:>10.2f}% | "
        f"{s['advantage_pp']:>+9.2f} pp"
    )


# ============================================================
# SEEN / UNSEEN AGGREGATES
# ============================================================

seen_phases = [
    p
    for p in TEST_PHASES
    if p in TRAINED_PHASES
]


unseen_phases = [
    p
    for p in TEST_PHASES
    if p not in TRAINED_PHASES
]


def mean_metric(
    phases,
    metric,
):

    return float(
        np.mean(
            [
                summary[p][
                    metric
                ]
                for p
                in phases
            ]
        )
    )


seen_raw_gain = mean_metric(
    seen_phases,
    "raw_gain",
)


seen_world_gain = mean_metric(
    seen_phases,
    "world_gain",
)


unseen_raw_gain = mean_metric(
    unseen_phases,
    "raw_gain",
)


unseen_world_gain = mean_metric(
    unseen_phases,
    "world_gain",
)


seen_advantage = (
    seen_world_gain
    - seen_raw_gain
) * 100


unseen_advantage = (
    unseen_world_gain
    - unseen_raw_gain
) * 100


print()

print("=" * 120)
print("SEEN VS UNSEEN SUMMARY")
print("=" * 120)


print(
    "Seen phases:",
    seen_phases,
)


print(
    f"Seen Raw mean gain   : "
    f"{seen_raw_gain * 100:.2f}%"
)


print(
    f"Seen World mean gain : "
    f"{seen_world_gain * 100:.2f}%"
)


print(
    f"Seen World advantage : "
    f"{seen_advantage:+.2f} pp"
)


print()


print(
    "Unseen phases:",
    unseen_phases,
)


print(
    f"Unseen Raw mean gain   : "
    f"{unseen_raw_gain * 100:.2f}%"
)


print(
    f"Unseen World mean gain : "
    f"{unseen_world_gain * 100:.2f}%"
)


print(
    f"Unseen World advantage : "
    f"{unseen_advantage:+.2f} pp"
)


# ============================================================
# CONSISTENCY
# ============================================================

raw_gains = np.array(
    [
        summary[p][
            "raw_gain"
        ]
        for p
        in TEST_PHASES
    ]
)


world_gains = np.array(
    [
        summary[p][
            "world_gain"
        ]
        for p
        in TEST_PHASES
    ]
)


print()

print("=" * 120)
print("PHASE CONSISTENCY")
print("=" * 120)


print(
    f"Raw gain mean ± std   : "
    f"{raw_gains.mean() * 100:.2f}% "
    f"± "
    f"{raw_gains.std() * 100:.2f} pp"
)


print(
    f"World gain mean ± std : "
    f"{world_gains.mean() * 100:.2f}% "
    f"± "
    f"{world_gains.std() * 100:.2f} pp"
)


# ============================================================
# SIMPLE DIAGNOSTIC
# ============================================================

print()

print("=" * 120)
print("DIAGNOSTIC")
print("=" * 120)


if (
    unseen_world_gain
    > unseen_raw_gain
):

    print(
        "PASS:"
    )

    print(
        "World representation has higher "
        "normalized temporal predictive gain "
        "on unseen tokenizer phases."
    )

else:

    print(
        "WARNING:"
    )

    print(
        "World representation does not "
        "outperform Raw on average across "
        "unseen tokenizer phases."
    )


if (
    unseen_advantage
    > 0
):

    print(
        f"Unseen-phase World advantage: "
        f"{unseen_advantage:+.2f} percentage points."
    )


print()

print("=" * 120)
print("DONE")
print("=" * 120)