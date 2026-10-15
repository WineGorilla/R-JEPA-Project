import numpy as np
import tensorflow_datasets as tfds

import torch
import torch.nn.functional as F

from factorizer import (
    WorldGridFactorizer,
)

from recomposer import (
    WorldGridRecomposer,
    reconstruction_loss,
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


# ------------------------------------------------------------
# Evaluation set
#
# Use validation clips that were NOT used during training.
#
# Training script used:
# validation[:8]
#
# So here we use:
# validation[8:40]
# ------------------------------------------------------------

VAL_START = 8

NUM_EVAL_CLIPS = 32


# ------------------------------------------------------------
# V-JEPA config
# ------------------------------------------------------------

NUM_FRAMES = 16

IMAGE_SIZE = 384

PATCH_SIZE = 16

TUBELET_SIZE = 2


# ------------------------------------------------------------
# Same phases used during training
# ------------------------------------------------------------

PHASES = [
    0,
    4,
    8,
    12,
]


# ------------------------------------------------------------
# Ignore border latent tokens
#
# Same logic as previous phase experiments.
# ------------------------------------------------------------

BORDER = 2


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

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
# 2. Load frozen V-JEPA
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
# 3. Load checkpoint
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
# 4. Build trained modules
# ============================================================

factorizer = WorldGridFactorizer(

    input_dim=768,

    hidden_dim=768,

    world_dim=WORLD_DIM,

    grid_dim=GRID_DIM,

    num_grid_classes=len(
        PHASES
    ),

).to(
    DEVICE
)


recomposer = WorldGridRecomposer(

    world_dim=WORLD_DIM,

    grid_dim=GRID_DIM,

    hidden_dim=768,

    output_dim=768,

).to(
    DEVICE
)


factorizer.load_state_dict(
    checkpoint[
        "factorizer"
    ]
)


recomposer.load_state_dict(
    checkpoint[
        "recomposer"
    ]
)


factorizer.eval()

recomposer.eval()


print(
    "Checkpoint loaded successfully."
)


# ============================================================
# 5. Load MOVi-C
# ============================================================

print()

print("=" * 80)

print(
    "Loading MOVi-C evaluation split"
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
    VAL_START + NUM_EVAL_CLIPS - 1,
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
# 7. Shift clip
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
        N
        == expected_tokens
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
# 9. Latent alignment
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
# 10. Cosine delta
# ============================================================

def cosine_delta(
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
        1.0
        - cosine
    )


    return (
        delta.mean().item()
    )


# ============================================================
# 11. Storage
# ============================================================

raw_results = {
    4: [],
    8: [],
    12: [],
}


world_results = {
    4: [],
    8: [],
    12: [],
}


recon_results = []


cross_results = []


grid_correct = 0

grid_total = 0


# ============================================================
# 12. Evaluation loop
# ============================================================

print()

print("=" * 80)

print(
    "Starting evaluation"
)

print("=" * 80)


with torch.no_grad():

    for clip_idx, sample in enumerate(
        eval_ds
    ):

        # ----------------------------------------------------
        # Original video
        # ----------------------------------------------------

        video = prepare_video(
            sample
        )


        video = video.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Build phase latents
        # ----------------------------------------------------

        z_list = []


        for phase in PHASES:

            shifted_video = shift_clip(
                video,
                dx=phase,
            )


            z = encode_video(
                shifted_video
            )


            z = latent_resample(
                z,
                dx_pixels=phase,
            )


            z_list.append(
                z
            )


        # ----------------------------------------------------
        # Factorize
        # ----------------------------------------------------

        worlds = []

        grids = []

        logits = []


        for z in z_list:

            (
                world,
                grid,
                grid_logits,
            ) = factorizer(
                z
            )


            worlds.append(
                world
            )


            grids.append(
                grid
            )


            logits.append(
                grid_logits
            )


        # ====================================================
        # A. Raw V-JEPA phase sensitivity
        # ====================================================

        raw_4 = cosine_delta(
            z_list[0],
            z_list[1],
        )


        raw_8 = cosine_delta(
            z_list[0],
            z_list[2],
        )


        raw_12 = cosine_delta(
            z_list[0],
            z_list[3],
        )


        raw_results[
            4
        ].append(
            raw_4
        )


        raw_results[
            8
        ].append(
            raw_8
        )


        raw_results[
            12
        ].append(
            raw_12
        )


        # ====================================================
        # B. World phase sensitivity
        # ====================================================

        world_4 = cosine_delta(
            worlds[0],
            worlds[1],
        )


        world_8 = cosine_delta(
            worlds[0],
            worlds[2],
        )


        world_12 = cosine_delta(
            worlds[0],
            worlds[3],
        )


        world_results[
            4
        ].append(
            world_4
        )


        world_results[
            8
        ].append(
            world_8
        )


        world_results[
            12
        ].append(
            world_12
        )


        # ====================================================
        # C. Same-phase reconstruction
        # ====================================================

        same_phase_losses = []


        for (
            world,
            grid,
            target,
        ) in zip(
            worlds,
            grids,
            z_list,
        ):

            pred = recomposer(
                world,
                grid,
            )


            loss = reconstruction_loss(
                pred,
                target,
            )


            same_phase_losses.append(
                loss.item()
            )


        recon_results.append(

            np.mean(
                same_phase_losses
            )
        )


        # ====================================================
        # D. Cross recomposition
        # ====================================================

        cross_losses = []


        # reference World + other Grid

        for i in range(
            1,
            len(PHASES),
        ):

            pred = recomposer(

                worlds[0],

                grids[i],
            )


            loss = reconstruction_loss(

                pred,

                z_list[i],
            )


            cross_losses.append(
                loss.item()
            )


            # ----------------------------------------------
            # Other World + reference Grid
            # ----------------------------------------------

            pred_ref = recomposer(

                worlds[i],

                grids[0],
            )


            loss_ref = reconstruction_loss(

                pred_ref,

                z_list[0],
            )


            cross_losses.append(
                loss_ref.item()
            )


        cross_results.append(

            np.mean(
                cross_losses
            )
        )


        # ====================================================
        # E. Grid phase classification
        # ====================================================

        for phase_idx, phase_logits in enumerate(
            logits
        ):

            pred = (
                phase_logits
                .argmax(
                    dim=-1
                )
            )


            grid_correct += (

                pred
                == phase_idx

            ).sum().item()


            grid_total += (
                pred.numel()
            )


        # ====================================================
        # Per-clip print
        # ====================================================

        print(

            f"clip={clip_idx:03d}"

            f" | raw@8="
            f"{raw_8:.6f}"

            f" | world@8="
            f"{world_8:.6f}"

            f" | reduction="
            f"{(1.0 - world_8 / raw_8) * 100:.1f}%"

            if raw_8 > 0

            else ""
        )


# ============================================================
# 13. Aggregate
# ============================================================

print()

print("=" * 100)

print(
    "PHASE SENSITIVITY: "
    "ORIGINAL V-JEPA vs WORLD BRANCH"
)

print("=" * 100)


print(

    f"{'Shift':>8}"

    f" | "

    f"{'V-JEPA Δ':>12}"

    f" | "

    f"{'World Δ':>12}"

    f" | "

    f"{'Reduction':>12}"
)


print("-" * 100)


summary = {}


for shift in [
    4,
    8,
    12,
]:

    raw = float(
        np.mean(
            raw_results[
                shift
            ]
        )
    )


    world = float(
        np.mean(
            world_results[
                shift
            ]
        )
    )


    if raw > 0:

        reduction = (

            1.0

            - world / raw
        )

    else:

        reduction = 0.0


    summary[
        shift
    ] = {

        "raw":
            raw,

        "world":
            world,

        "reduction":
            reduction,
    }


    print(

        f"{shift:8d}"

        f" | "

        f"{raw:12.6f}"

        f" | "

        f"{world:12.6f}"

        f" | "

        f"{reduction * 100:11.2f}%"
    )


# ============================================================
# 14. Other metrics
# ============================================================

grid_accuracy = (

    grid_correct

    / grid_total
)


mean_recon = float(
    np.mean(
        recon_results
    )
)


mean_cross = float(
    np.mean(
        cross_results
    )
)


print()

print("=" * 80)

print(
    "DISENTANGLEMENT / RECONSTRUCTION METRICS"
)

print("=" * 80)


print(
    f"Grid classification accuracy : "
    f"{grid_accuracy:.4f}"
)


print(
    f"Same-phase reconstruction    : "
    f"{mean_recon:.6f}"
)


print(
    f"Cross recomposition          : "
    f"{mean_cross:.6f}"
)


# ============================================================
# 15. Main 8px result
# ============================================================

raw_8 = summary[
    8
]["raw"]


world_8 = summary[
    8
]["world"]


reduction_8 = summary[
    8
]["reduction"]


print()

print("=" * 80)

print(
    "MAIN HALF-PATCH RESULT"
)

print("=" * 80)


print(
    f"Original V-JEPA Δ@8 : "
    f"{raw_8:.6f}"
)


print(
    f"World branch Δ@8    : "
    f"{world_8:.6f}"
)


print(
    f"Relative reduction  : "
    f"{reduction_8 * 100:.2f}%"
)


# ============================================================
# 16. Interpretation helper
# ============================================================

print()

print("=" * 80)

print(
    "INTERPRETATION"
)

print("=" * 80)


print(
    """
The comparison uses the SAME:

    clips
    shifts
    alignment
    border crop
    cosine-distance metric

for both:

    Original V-JEPA latent
    World branch latent

Therefore the V-JEPA-vs-World comparison is much
cleaner than comparing values from separate experiments.

Positive evidence should look like:

    World Δ@4  < V-JEPA Δ@4
    World Δ@8  < V-JEPA Δ@8
    World Δ@12 < V-JEPA Δ@12

while:

    Grid classification accuracy > 0.25

and reconstruction remains reasonably low.

This would indicate that the learned World branch
is more invariant to patch-grid phase while the
Grid branch retains information about the tokenizer
intervention.
"""
)


print("=" * 80)

print(
    "DONE"
)

print("=" * 80)