import itertools
import numpy as np
import tensorflow_datasets as tfds

import torch
import torch.nn.functional as F

from factorizer import (
    WorldGridFactorizer,
)


# ============================================================
# 1. Config
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


VJEPA_REPO = (
    "/data/mi2-interns/ruiyu/"
    "R-JEPA/vjepa2"
)


CHECKPOINT_PATH = (
    "/data/mi2-interns/ruiyu/"
    "R-JEPA/checkpoints/phase1/"
    "best.pt"
)


GCS_DATA_DIR = (
    "gs://kubric-public/tfds"
)


# ============================================================
# Completely unseen validation clips
#
# Training:
#     train[:64]
#
# Model selection:
#     validation[:8]
#
# Previous evaluation:
#     validation[8:40]
#
# This experiment:
#     validation[40:72]
# ============================================================

VAL_START = 40

NUM_EVAL_CLIPS = 32


# ============================================================
# V-JEPA config
# ============================================================

NUM_FRAMES = 16

IMAGE_SIZE = 384

PATCH_SIZE = 16

TUBELET_SIZE = 2


# ============================================================
# Main intervention
#
# 8 px = half patch
# This was the strongest phase intervention.
# ============================================================

PHASE_REFERENCE = 0

PHASE_SHIFT = 8


# ============================================================
# Border crop
# ============================================================

BORDER = 2


# ============================================================
# Normalization
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
# 2. Load V-JEPA
# ============================================================

print("=" * 80)

print(
    "Loading frozen "
    "V-JEPA 2.1 ViT-B/16"
)

print("=" * 80)


encoder, predictor = torch.hub.load(
    VJEPA_REPO,
    "vjepa2_1_vit_base_384",
    source="local",
)


encoder = encoder.to(
    DEVICE
)


encoder.eval()


del predictor


for p in encoder.parameters():

    p.requires_grad = False


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
# 3. Load Phase-1 checkpoint
# ============================================================

print()

print("=" * 80)

print(
    "Loading Phase-1 checkpoint"
)

print("=" * 80)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
)


config = checkpoint[
    "config"
]


WORLD_DIM = config[
    "world_dim"
]


GRID_DIM = config[
    "grid_dim"
]


print(
    "Checkpoint epoch:",
    checkpoint[
        "epoch"
    ],
)


print(
    "World dim:",
    WORLD_DIM,
)


print(
    "Grid dim:",
    GRID_DIM,
)


# ============================================================
# 4. Build trained Factorizer
# ============================================================

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
    checkpoint[
        "factorizer"
    ]
)


factorizer.eval()


for p in factorizer.parameters():

    p.requires_grad = False


print(
    "Factorizer loaded successfully."
)


# ============================================================
# 5. Load unseen MOVi-C clips
# ============================================================

print()

print("=" * 80)

print(
    "Loading unseen MOVi-C clips"
)

print("=" * 80)


builder = tfds.builder(

    "movi_c/256x256",

    data_dir=GCS_DATA_DIR,
)


eval_ds = builder.as_dataset(

    split=(
        f"validation["
        f"{VAL_START}:"
        f"{VAL_START + NUM_EVAL_CLIPS}]"
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
    "Validation range:",
    VAL_START,
    "to",
    VAL_START
    + NUM_EVAL_CLIPS
    - 1,
)


# ============================================================
# 6. Prepare video
# ============================================================

def prepare_video(sample):

    video = (
        sample["video"]
        .numpy()
        [:NUM_FRAMES]
    )


    # [T,H,W,C]
    # ->
    # [1,C,T,H,W]

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


    # --------------------------------------------------------
    # Resize 256 -> 384
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


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

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
# 7. Shift video
# ============================================================

def shift_clip(
    video,
    dx,
    dy=0,
):

    if (
        dx == 0
        and dy == 0
    ):

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
                -2.0
                * dy
                / (H - 1),
            ],
        ],

        dtype=x.dtype,

        device=x.device,
    )


    theta = theta.unsqueeze(0)


    theta = theta.repeat(
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
# 8. Encode V-JEPA
# ============================================================

def encode_video(
    video,
):

    with torch.no_grad():

        z = encoder(
            video
        )


    z = z.detach()


    B, N, D = (
        z.shape
    )


    temporal_tokens = (
        NUM_FRAMES
        // TUBELET_SIZE
    )


    spatial_tokens = (
        IMAGE_SIZE
        // PATCH_SIZE
    )


    expected_tokens = (

        temporal_tokens

        *

        spatial_tokens

        *

        spatial_tokens
    )


    assert (
        N == expected_tokens
    ), (
        f"Expected "
        f"{expected_tokens}, "
        f"got {N}"
    )


    z = z.reshape(

        B,

        temporal_tokens,

        spatial_tokens,

        spatial_tokens,

        D,
    )


    return z


# ============================================================
# 9. Align latent back to world coordinates
# ============================================================

def latent_resample(
    latent,
    dx_pixels,
    dy_pixels=0,
):

    if (
        dx_pixels == 0
        and dy_pixels == 0
    ):

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


    dy_tokens = (
        dy_pixels
        / PATCH_SIZE
    )


    shift_x_norm = (

        2.0

        * dx_tokens

        / (W - 1)
    )


    shift_y_norm = (

        2.0

        * dy_tokens

        / (H - 1)
    )


    ys, xs = torch.meshgrid(

        torch.linspace(

            -1,
            1,
            H,

            device=x.device,

            dtype=x.dtype,
        ),

        torch.linspace(

            -1,
            1,
            W,

            device=x.device,

            dtype=x.dtype,
        ),

        indexing="ij",
    )


    grid = torch.stack(

        [
            xs + shift_x_norm,
            ys + shift_y_norm,
        ],

        dim=-1,
    )


    grid = grid.unsqueeze(0)


    grid = grid.repeat(
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
# 10. Token-level cosine distance
# ============================================================

def token_cosine_delta(
    a,
    b,
    border=BORDER,
):

    if border > 0:

        a = a[
            :,
            :,
            border:-border,
            border:-border,
            :,
        ]


        b = b[
            :,
            :,
            border:-border,
            border:-border,
            :,
        ]


    a = F.normalize(
        a,
        dim=-1,
    )


    b = F.normalize(
        b,
        dim=-1,
    )


    cosine = (
        a * b
    ).sum(
        dim=-1
    )


    delta = (
        1.0 - cosine
    )


    return (
        delta.mean().item()
    )


# ============================================================
# 11. Global pooled representation
#
# Different videos do NOT have exact token-to-token
# correspondence.
#
# Therefore content preservation is measured using
# one pooled vector per video.
# ============================================================

def global_pool(
    features,
    border=BORDER,
):

    if border > 0:

        features = features[
            :,
            :,
            border:-border,
            border:-border,
            :,
        ]


    pooled = features.mean(
        dim=(
            1,
            2,
            3,
        )
    )


    pooled = F.normalize(
        pooled,
        dim=-1,
    )


    return pooled


# ============================================================
# 12. Vector cosine distance
# ============================================================

def vector_cosine_delta(
    a,
    b,
):

    a = F.normalize(
        a,
        dim=-1,
    )


    b = F.normalize(
        b,
        dim=-1,
    )


    cosine = (
        a * b
    ).sum(
        dim=-1
    )


    return (
        1.0
        - cosine
    )


# ============================================================
# 13. Storage
# ============================================================

raw_phase_token = []

world_phase_token = []


raw_phase_global = []

world_phase_global = []


# Phase-0 pooled representation
# for between-video comparison.

raw_reference_vectors = []

world_reference_vectors = []


# Also save phase-8 vectors
# to check representation consistency.

raw_shift_vectors = []

world_shift_vectors = []


# ============================================================
# 14. Encode all clips
# ============================================================

print()

print("=" * 80)

print(
    "Encoding evaluation clips"
)

print("=" * 80)


with torch.no_grad():

    for clip_idx, sample in enumerate(
        eval_ds
    ):

        # ====================================================
        # Original clip
        # ====================================================

        video = prepare_video(
            sample
        )


        video = video.to(
            DEVICE
        )


        # ====================================================
        # Phase 0
        # ====================================================

        z0 = encode_video(
            video
        )


        (
            world0,
            grid0,
            logits0,
        ) = factorizer(
            z0
        )


        # ====================================================
        # Phase 8
        # ====================================================

        video8 = shift_clip(
            video,
            dx=PHASE_SHIFT,
        )


        z8 = encode_video(
            video8
        )


        z8 = latent_resample(
            z8,
            dx_pixels=PHASE_SHIFT,
        )


        (
            world8,
            grid8,
            logits8,
        ) = factorizer(
            z8
        )


        # ====================================================
        # A. Token-level same-video phase sensitivity
        # ====================================================

        raw_token_delta = (
            token_cosine_delta(
                z0,
                z8,
            )
        )


        world_token_delta = (
            token_cosine_delta(
                world0,
                world8,
            )
        )


        raw_phase_token.append(
            raw_token_delta
        )


        world_phase_token.append(
            world_token_delta
        )


        # ====================================================
        # B. Global pooled representations
        # ====================================================

        raw0_global = global_pool(
            z0
        )


        raw8_global = global_pool(
            z8
        )


        world0_global = global_pool(
            world0
        )


        world8_global = global_pool(
            world8
        )


        # ====================================================
        # C. Same-video phase distance
        # ====================================================

        raw_global_delta = (
            vector_cosine_delta(
                raw0_global,
                raw8_global,
            ).item()
        )


        world_global_delta = (
            vector_cosine_delta(
                world0_global,
                world8_global,
            ).item()
        )


        raw_phase_global.append(
            raw_global_delta
        )


        world_phase_global.append(
            world_global_delta
        )


        # ====================================================
        # D. Store phase-0 vector for content comparison
        # ====================================================

        raw_reference_vectors.append(
            raw0_global.cpu()
        )


        world_reference_vectors.append(
            world0_global.cpu()
        )


        raw_shift_vectors.append(
            raw8_global.cpu()
        )


        world_shift_vectors.append(
            world8_global.cpu()
        )


        print(

            f"clip={clip_idx:03d}"

            f" | raw token phase="
            f"{raw_token_delta:.6f}"

            f" | world token phase="
            f"{world_token_delta:.6f}"

            f" | raw global phase="
            f"{raw_global_delta:.6f}"

            f" | world global phase="
            f"{world_global_delta:.6f}"
        )


# ============================================================
# 15. Stack global representations
# ============================================================

raw_reference_vectors = torch.cat(
    raw_reference_vectors,
    dim=0,
)


world_reference_vectors = torch.cat(
    world_reference_vectors,
    dim=0,
)


raw_shift_vectors = torch.cat(
    raw_shift_vectors,
    dim=0,
)


world_shift_vectors = torch.cat(
    world_shift_vectors,
    dim=0,
)


# ============================================================
# 16. Between-video distances
#
# Use ALL unordered clip pairs.
#
# 32 clips -> 496 pairs.
# ============================================================

raw_between = []

world_between = []


raw_between_shift = []

world_between_shift = []


pairs = list(
    itertools.combinations(
        range(
            NUM_EVAL_CLIPS
        ),
        2,
    )
)


for i, j in pairs:

    # --------------------------------------------------------
    # Phase-0 content distance
    # --------------------------------------------------------

    raw_delta = (
        vector_cosine_delta(

            raw_reference_vectors[
                i:i+1
            ],

            raw_reference_vectors[
                j:j+1
            ],
        ).item()
    )


    world_delta = (
        vector_cosine_delta(

            world_reference_vectors[
                i:i+1
            ],

            world_reference_vectors[
                j:j+1
            ],
        ).item()
    )


    raw_between.append(
        raw_delta
    )


    world_between.append(
        world_delta
    )


    # --------------------------------------------------------
    # Same test under phase 8
    # --------------------------------------------------------

    raw_delta_shift = (
        vector_cosine_delta(

            raw_shift_vectors[
                i:i+1
            ],

            raw_shift_vectors[
                j:j+1
            ],
        ).item()
    )


    world_delta_shift = (
        vector_cosine_delta(

            world_shift_vectors[
                i:i+1
            ],

            world_shift_vectors[
                j:j+1
            ],
        ).item()
    )


    raw_between_shift.append(
        raw_delta_shift
    )


    world_between_shift.append(
        world_delta_shift
    )


# ============================================================
# 17. Statistics helper
# ============================================================

def summarize(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )


    return {

        "mean":
            float(
                np.mean(values)
            ),

        "std":
            float(
                np.std(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "p10":
            float(
                np.quantile(
                    values,
                    0.10,
                )
            ),

        "p90":
            float(
                np.quantile(
                    values,
                    0.90,
                )
            ),
    }


# ============================================================
# 18. Aggregate statistics
# ============================================================

raw_phase_token_stats = summarize(
    raw_phase_token
)


world_phase_token_stats = summarize(
    world_phase_token
)


raw_phase_global_stats = summarize(
    raw_phase_global
)


world_phase_global_stats = summarize(
    world_phase_global
)


raw_between_stats = summarize(
    raw_between
)


world_between_stats = summarize(
    world_between
)


raw_between_shift_stats = summarize(
    raw_between_shift
)


world_between_shift_stats = summarize(
    world_between_shift
)


# ============================================================
# 19. Token-level phase sensitivity
# ============================================================

print()

print("=" * 90)

print(
    "A. TOKEN-LEVEL PHASE SENSITIVITY"
)

print("=" * 90)


print(

    f"{'Representation':>20}"

    f" | "

    f"{'Mean Δ':>12}"

    f" | "

    f"{'Std':>12}"
)


print("-" * 90)


print(

    f"{'Original V-JEPA':>20}"

    f" | "

    f"{raw_phase_token_stats['mean']:12.6f}"

    f" | "

    f"{raw_phase_token_stats['std']:12.6f}"
)


print(

    f"{'World branch':>20}"

    f" | "

    f"{world_phase_token_stats['mean']:12.6f}"

    f" | "

    f"{world_phase_token_stats['std']:12.6f}"
)


token_reduction = (

    1.0

    -

    world_phase_token_stats[
        "mean"
    ]

    /

    raw_phase_token_stats[
        "mean"
    ]
)


print()


print(
    f"Token-level phase reduction: "
    f"{token_reduction * 100:.2f}%"
)


# ============================================================
# 20. Global same-video phase distance
# ============================================================

print()

print("=" * 90)

print(
    "B. SAME VIDEO, DIFFERENT PATCH PHASE"
)

print("=" * 90)


print(

    f"{'Representation':>20}"

    f" | "

    f"{'Mean Δ':>12}"

    f" | "

    f"{'Std':>12}"

    f" | "

    f"{'Median':>12}"
)


print("-" * 90)


print(

    f"{'Original V-JEPA':>20}"

    f" | "

    f"{raw_phase_global_stats['mean']:12.6f}"

    f" | "

    f"{raw_phase_global_stats['std']:12.6f}"

    f" | "

    f"{raw_phase_global_stats['median']:12.6f}"
)


print(

    f"{'World branch':>20}"

    f" | "

    f"{world_phase_global_stats['mean']:12.6f}"

    f" | "

    f"{world_phase_global_stats['std']:12.6f}"

    f" | "

    f"{world_phase_global_stats['median']:12.6f}"
)


# ============================================================
# 21. Different-video content distance
# ============================================================

print()

print("=" * 90)

print(
    "C. DIFFERENT VIDEOS, SAME PATCH PHASE"
)

print("=" * 90)


print(
    "All unordered clip pairs:",
    len(pairs),
)


print()


print(

    f"{'Representation':>20}"

    f" | "

    f"{'Mean Δ':>12}"

    f" | "

    f"{'Std':>12}"

    f" | "

    f"{'Median':>12}"
)


print("-" * 90)


print(

    f"{'Original V-JEPA':>20}"

    f" | "

    f"{raw_between_stats['mean']:12.6f}"

    f" | "

    f"{raw_between_stats['std']:12.6f}"

    f" | "

    f"{raw_between_stats['median']:12.6f}"
)


print(

    f"{'World branch':>20}"

    f" | "

    f"{world_between_stats['mean']:12.6f}"

    f" | "

    f"{world_between_stats['std']:12.6f}"

    f" | "

    f"{world_between_stats['median']:12.6f}"
)


# ============================================================
# 22. Separation ratios
# ============================================================

raw_ratio = (

    raw_between_stats[
        "mean"
    ]

    /

    raw_phase_global_stats[
        "mean"
    ]
)


world_ratio = (

    world_between_stats[
        "mean"
    ]

    /

    world_phase_global_stats[
        "mean"
    ]
)


ratio_improvement = (

    world_ratio

    /

    raw_ratio
)


print()

print("=" * 90)

print(
    "D. CONTENT / PHASE SEPARATION RATIO"
)

print("=" * 90)


print(
    "Definition:"
)


print()


print(
    "    separation ratio"
)


print(
    "        ="
)


print(
    "    different-video distance"
)


print(
    "        /"
)


print(
    "    same-video phase distance"
)


print()


print(
    f"Original V-JEPA ratio : "
    f"{raw_ratio:.3f}x"
)


print(
    f"World branch ratio    : "
    f"{world_ratio:.3f}x"
)


print(
    f"Ratio improvement     : "
    f"{ratio_improvement:.3f}x"
)


# ============================================================
# 23. Content retention
# ============================================================

content_retention = (

    world_between_stats[
        "mean"
    ]

    /

    raw_between_stats[
        "mean"
    ]
)


print()

print("=" * 90)

print(
    "E. DIFFERENT-VIDEO DISTANCE RETENTION"
)

print("=" * 90)


print(
    f"Raw different-video Δ   : "
    f"{raw_between_stats['mean']:.6f}"
)


print(
    f"World different-video Δ : "
    f"{world_between_stats['mean']:.6f}"
)


print(
    f"Relative distance retention: "
    f"{content_retention * 100:.2f}%"
)


# ============================================================
# 24. Check if content separation survives phase change
# ============================================================

print()

print("=" * 90)

print(
    "F. DIFFERENT-VIDEO DISTANCE UNDER 8PX PHASE"
)

print("=" * 90)


print(
    f"Raw phase-0 inter-video Δ : "
    f"{raw_between_stats['mean']:.6f}"
)


print(
    f"Raw phase-8 inter-video Δ : "
    f"{raw_between_shift_stats['mean']:.6f}"
)


print()


print(
    f"World phase-0 inter-video Δ : "
    f"{world_between_stats['mean']:.6f}"
)


print(
    f"World phase-8 inter-video Δ : "
    f"{world_between_shift_stats['mean']:.6f}"
)


# ============================================================
# 25. Main conclusion summary
# ============================================================

print()

print("=" * 90)

print(
    "MAIN CONTENT-PRESERVATION SUMMARY"
)

print("=" * 90)


print(
    f"V-JEPA token phase Δ@8 : "
    f"{raw_phase_token_stats['mean']:.6f}"
)


print(
    f"World token phase Δ@8  : "
    f"{world_phase_token_stats['mean']:.6f}"
)


print(
    f"Phase sensitivity reduction: "
    f"{token_reduction * 100:.2f}%"
)


print()


print(
    f"V-JEPA global intra-video phase Δ : "
    f"{raw_phase_global_stats['mean']:.6f}"
)


print(
    f"World global intra-video phase Δ  : "
    f"{world_phase_global_stats['mean']:.6f}"
)


print()


print(
    f"V-JEPA inter-video Δ : "
    f"{raw_between_stats['mean']:.6f}"
)


print(
    f"World inter-video Δ  : "
    f"{world_between_stats['mean']:.6f}"
)


print()


print(
    f"V-JEPA separation ratio : "
    f"{raw_ratio:.3f}x"
)


print(
    f"World separation ratio  : "
    f"{world_ratio:.3f}x"
)


print(
    f"Separation improvement  : "
    f"{ratio_improvement:.3f}x"
)


# ============================================================
# 26. Automatic diagnostic
# ============================================================

print()

print("=" * 90)

print(
    "AUTOMATIC DIAGNOSTIC"
)

print("=" * 90)


if (
    world_phase_global_stats["mean"]
    < raw_phase_global_stats["mean"]
):

    print(
        "[PASS] World branch is more invariant "
        "to patch-grid phase."
    )

else:

    print(
        "[FAIL] World branch did not reduce "
        "global phase sensitivity."
    )


if (
    world_between_stats["mean"]
    >
    world_phase_global_stats["mean"]
):

    print(
        "[PASS] Different videos remain farther apart "
        "than different phases of the same video."
    )

else:

    print(
        "[FAIL] Possible representation collapse."
    )


if (
    world_ratio
    >
    raw_ratio
):

    print(
        "[PASS] Content/phase separation improved."
    )

else:

    print(
        "[FAIL] Separation ratio did not improve."
    )


if (
    content_retention
    > 0.25
):

    print(
        "[PASS] World branch retains substantial "
        "between-video variation."
    )

else:

    print(
        "[WARNING] World branch may be over-compressing "
        "content variation."
    )


# ============================================================
# 27. Interpretation
# ============================================================

print()

print("=" * 90)

print(
    "INTERPRETATION"
)

print("=" * 90)


print(
    """
The purpose of this experiment is NOT simply to show
that World features have smaller distances.

A trivial smoothing or collapsed representation could
also make every distance small.

The important comparison is:

    SAME video
    + different tokenizer phase

versus

    DIFFERENT videos
    + same tokenizer phase

A successful World representation should satisfy:

    D(World(x, phase0), World(x, phase8))
        <<

    D(World(x_i, phase0), World(x_j, phase0))

for i != j.

Therefore the key metric is:

    Separation Ratio
        =
    Between-video distance
        /
    Same-video phase distance

If the World branch has a substantially larger
Separation Ratio than the original V-JEPA latent,
this supports the claim that the module selectively
suppresses tokenizer-phase variation rather than
simply collapsing all representations.
"""
)


print("=" * 90)

print(
    "DONE"
)

print("=" * 90)