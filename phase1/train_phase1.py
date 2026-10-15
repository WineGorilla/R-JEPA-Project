import random
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from factorizer import (
    WorldGridFactorizer,
    world_consistency_loss,
    world_variance_loss,
)

from recomposer import (
    WorldGridRecomposer,
    reconstruction_loss,
    cross_recomposition_loss,
)


# ============================================================
# 1. Config
# ============================================================

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

VJEPA_REPO = (
    "/data/mi2-interns/ruiyu/"
    "R-JEPA/vjepa2"
)

GCS_DATA_DIR = (
    "gs://kubric-public/tfds"
)

OUTPUT_DIR = Path(
    "/data/mi2-interns/ruiyu/"
    "R-JEPA/checkpoints/phase1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Video / V-JEPA
# ------------------------------------------------------------

NUM_FRAMES = 16

IMAGE_SIZE = 384

PATCH_SIZE = 16

TUBELET_SIZE = 2


# ------------------------------------------------------------
# Grid interventions
#
# Patch size = 16
#
# We use four phases:
#
# 0 / 16
# 4 / 16
# 8 / 16
# 12 / 16
#
# 16px is equivalent phase to 0px,
# therefore it is NOT treated as a new class.
# ------------------------------------------------------------

PHASES = [
    0,
    4,
    8,
    12,
]


# ------------------------------------------------------------
# Pilot training size
# ------------------------------------------------------------

NUM_TRAIN_CLIPS = 64

NUM_VAL_CLIPS = 8

EPOCHS = 5


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

LR = 1e-4

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0


# ------------------------------------------------------------
# Factor dimensions
# ------------------------------------------------------------

WORLD_DIM = 512

GRID_DIM = 128


# ------------------------------------------------------------
# Loss weights
# ------------------------------------------------------------

LAMBDA_WORLD = 1.0

LAMBDA_GRID = 0.25

LAMBDA_RECON = 1.0

LAMBDA_CROSS = 1.0

LAMBDA_VAR = 0.05


# ------------------------------------------------------------
# ImageNet normalization
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
# 2. Seed
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# 3. Load frozen V-JEPA
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


# Predictor is not used in Phase 1.
del predictor


# Freeze V-JEPA.
for p in encoder.parameters():

    p.requires_grad = False


print(
    "Device:",
    DEVICE,
)


print(
    "GPU:",
    (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    ),
)


print(
    "Encoder params:",
    sum(
        p.numel()
        for p in encoder.parameters()
    ) / 1e6,
    "M",
)


# ============================================================
# 4. Trainable modules
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


trainable_params = (

    list(
        factorizer.parameters()
    )

    +

    list(
        recomposer.parameters()
    )
)


num_trainable = sum(

    p.numel()

    for p in trainable_params

    if p.requires_grad
)


print(
    "Trainable new-module params:",
    num_trainable / 1e6,
    "M",
)


optimizer = AdamW(

    trainable_params,

    lr=LR,

    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 5. Load MOVi-C
# ============================================================

print()

print("=" * 80)

print(
    "Loading MOVi-C"
)

print("=" * 80)


builder = tfds.builder(

    "movi_c/256x256",

    data_dir=GCS_DATA_DIR,
)


train_ds = builder.as_dataset(

    split=(
        f"train[:"
        f"{NUM_TRAIN_CLIPS}]"
    ),

    shuffle_files=False,
)


val_ds = builder.as_dataset(

    split=(
        f"validation[:"
        f"{NUM_VAL_CLIPS}]"
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
# 6. Prepare video
# ============================================================

def prepare_video(sample):

    # --------------------------------------------------------
    # MOVi-C:
    #
    # [24, 256, 256, 3]
    #
    # use first 16 frames
    # --------------------------------------------------------

    video = (
        sample["video"]
        .numpy()
        [:NUM_FRAMES]
    )


    # --------------------------------------------------------
    # numpy -> torch
    #
    # [T,H,W,C]
    #
    # ->
    #
    # [1,C,T,H,W]
    # --------------------------------------------------------

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
    # Resize:
    #
    # 256x256
    # ->
    # 384x384
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
    # ImageNet normalization
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
# 7. Shift complete video clip
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
# 8. Frozen V-JEPA encoder
# ============================================================

def encode_video(video):

    # ========================================================
    # IMPORTANT FIX
    #
    # DO NOT use torch.inference_mode() here.
    #
    # V-JEPA itself is frozen, so no_grad() is enough.
    #
    # The resulting latent can then be passed to Factorizer,
    # whose parameters DO require gradients.
    # ========================================================

    with torch.no_grad():

        z = encoder(
            video
        )


    # --------------------------------------------------------
    # Explicitly detach from frozen V-JEPA graph.
    #
    # This is a NORMAL PyTorch tensor,
    # not an inference tensor.
    # --------------------------------------------------------

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
    ), (
        f"Expected "
        f"{expected_tokens} tokens, "
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


    # --------------------------------------------------------
    # Pixel displacement
    # ->
    # latent token displacement
    # --------------------------------------------------------

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
# 10. Produce all phase representations
# ============================================================

def build_phase_latents(
    video,
):

    z_list = []


    for phase in PHASES:

        # ----------------------------------------------------
        # Shift input
        # ----------------------------------------------------

        shifted_video = shift_clip(

            video,

            dx=phase,
        )


        # ----------------------------------------------------
        # Frozen V-JEPA
        # ----------------------------------------------------

        z = encode_video(
            shifted_video
        )


        # ----------------------------------------------------
        # Bring latent back into common
        # world coordinate system.
        # ----------------------------------------------------

        z = latent_resample(

            z,

            dx_pixels=phase,
        )


        z_list.append(
            z
        )


    return z_list


# ============================================================
# 11. Factorizer forward
# ============================================================

def run_factorizer(
    z_list,
):

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


    return (
        worlds,
        grids,
        logits,
    )


# ============================================================
# 12. All training losses
# ============================================================

def compute_losses(
    z_list,
):

    (
        worlds,
        grids,
        logits,
    ) = run_factorizer(
        z_list
    )


    # ========================================================
    # A. World consistency
    #
    # Different patch-grid phases
    # should share the same world representation.
    # ========================================================

    loss_world = (
        world_consistency_loss(
            worlds
        )
    )


    # ========================================================
    # B. Grid classification
    #
    # Grid branch must keep phase information.
    # ========================================================

    grid_losses = []


    for phase_idx, phase_logits in enumerate(
        logits
    ):

        labels = torch.full(

            (
                phase_logits.shape[0],
            ),

            phase_idx,

            dtype=torch.long,

            device=phase_logits.device,
        )


        phase_loss = F.cross_entropy(

            phase_logits,

            labels,
        )


        grid_losses.append(
            phase_loss
        )


    loss_grid = torch.stack(
        grid_losses
    ).mean()


    # ========================================================
    # C. Same-phase reconstruction
    #
    # R(W_g, G_g)
    # ->
    # Z_g
    # ========================================================

    recon_losses = []


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


        recon_losses.append(
            loss
        )


    loss_recon = torch.stack(
        recon_losses
    ).mean()


    # ========================================================
    # D. Cross-recomposition
    #
    # R(W_a, G_b)
    # ->
    # Z_b
    # ========================================================

    loss_cross = (
        cross_recomposition_loss(

            recomposer,

            worlds,

            grids,

            z_list,
        )
    )


    # ========================================================
    # E. Anti-collapse regularizer
    # ========================================================

    variance_losses = []


    for world in worlds:

        loss = (
            world_variance_loss(
                world
            )
        )


        variance_losses.append(
            loss
        )


    loss_var = torch.stack(
        variance_losses
    ).mean()


    # ========================================================
    # Total
    # ========================================================

    total = (

        LAMBDA_WORLD
        * loss_world

        +

        LAMBDA_GRID
        * loss_grid

        +

        LAMBDA_RECON
        * loss_recon

        +

        LAMBDA_CROSS
        * loss_cross

        +

        LAMBDA_VAR
        * loss_var
    )


    return {

        "total":
            total,

        "world":
            loss_world,

        "grid":
            loss_grid,

        "recon":
            loss_recon,

        "cross":
            loss_cross,

        "var":
            loss_var,

        "worlds":
            worlds,

        "grids":
            grids,

        "logits":
            logits,
    }


# ============================================================
# 13. World phase sensitivity
# ============================================================

@torch.no_grad()
def world_phase_delta(
    world_a,
    world_b,
):

    a = F.normalize(
        world_a,
        dim=-1,
    )


    b = F.normalize(
        world_b,
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
# 14. Validation
# ============================================================

@torch.no_grad()
def evaluate(dataset):

    factorizer.eval()

    recomposer.eval()


    meters = {

        "total": [],

        "world": [],

        "grid": [],

        "recon": [],

        "cross": [],

        "var": [],

        "delta_4": [],

        "delta_8": [],

        "delta_12": [],

        "grid_acc": [],
    }


    for sample in dataset:

        video = prepare_video(
            sample
        )


        video = video.to(
            DEVICE
        )


        z_list = (
            build_phase_latents(
                video
            )
        )


        outputs = (
            compute_losses(
                z_list
            )
        )


        # ----------------------------------------------------
        # Loss values
        # ----------------------------------------------------

        for key in [

            "total",

            "world",

            "grid",

            "recon",

            "cross",

            "var",
        ]:

            meters[
                key
            ].append(

                outputs[
                    key
                ].item()
            )


        worlds = (
            outputs[
                "worlds"
            ]
        )


        # ----------------------------------------------------
        # World phase sensitivity
        # ----------------------------------------------------

        meters[
            "delta_4"
        ].append(

            world_phase_delta(
                worlds[0],
                worlds[1],
            )
        )


        meters[
            "delta_8"
        ].append(

            world_phase_delta(
                worlds[0],
                worlds[2],
            )
        )


        meters[
            "delta_12"
        ].append(

            world_phase_delta(
                worlds[0],
                worlds[3],
            )
        )


        # ----------------------------------------------------
        # Grid classification accuracy
        # ----------------------------------------------------

        correct = 0

        count = 0


        for (
            phase_idx,
            phase_logits,
        ) in enumerate(
            outputs[
                "logits"
            ]
        ):

            pred = (
                phase_logits
                .argmax(
                    dim=-1
                )
            )


            correct += (
                pred
                == phase_idx
            ).sum().item()


            count += (
                pred.numel()
            )


        meters[
            "grid_acc"
        ].append(

            correct
            / count
        )


    result = {}


    for (
        key,
        values,
    ) in meters.items():

        result[
            key
        ] = float(
            np.mean(
                values
            )
        )


    return result


# ============================================================
# 15. Training
# ============================================================

print()

print("=" * 80)

print(
    "Starting Phase-1 training"
)

print("=" * 80)


best_val = float(
    "inf"
)


global_step = 0


for epoch in range(
    1,
    EPOCHS + 1,
):

    factorizer.train()

    recomposer.train()


    epoch_total_losses = []


    print()

    print(
        f"Epoch "
        f"{epoch}/{EPOCHS}"
    )


    # ========================================================
    # Training loop
    # ========================================================

    for (
        clip_idx,
        sample,
    ) in enumerate(
        train_ds
    ):

        # ----------------------------------------------------
        # MOVi-C clip
        # ----------------------------------------------------

        video = prepare_video(
            sample
        )


        video = video.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Frozen V-JEPA representations
        # ----------------------------------------------------

        z_list = (
            build_phase_latents(
                video
            )
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward trainable modules
        # ----------------------------------------------------

        outputs = (
            compute_losses(
                z_list
            )
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        outputs[
            "total"
        ].backward()


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(

            trainable_params,

            GRAD_CLIP,
        )


        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()


        epoch_total_losses.append(

            outputs[
                "total"
            ].item()
        )


        global_step += 1


        # ----------------------------------------------------
        # Monitor
        # ----------------------------------------------------

        if (
            clip_idx
            % 8
            == 0
        ):

            worlds = (
                outputs[
                    "worlds"
                ]
            )


            delta_8 = (
                world_phase_delta(

                    worlds[0],

                    worlds[2],
                )
            )


            print(

                f"clip="
                f"{clip_idx:03d}"

                f" | total="
                f"{outputs['total'].item():.4f}"

                f" | world="
                f"{outputs['world'].item():.4f}"

                f" | grid="
                f"{outputs['grid'].item():.4f}"

                f" | recon="
                f"{outputs['recon'].item():.4f}"

                f" | cross="
                f"{outputs['cross'].item():.4f}"

                f" | var="
                f"{outputs['var'].item():.4f}"

                f" | WΔ@8="
                f"{delta_8:.4f}"
            )


    # ========================================================
    # Epoch summary
    # ========================================================

    print()

    print(
        "Train total:",
        float(
            np.mean(
                epoch_total_losses
            )
        ),
    )


    # ========================================================
    # Validation
    # ========================================================

    val = evaluate(
        val_ds
    )


    print()

    print("=" * 60)

    print(
        "Validation"
    )

    print("=" * 60)


    print(
        f"total      : "
        f"{val['total']:.6f}"
    )


    print(
        f"world loss : "
        f"{val['world']:.6f}"
    )


    print(
        f"grid loss  : "
        f"{val['grid']:.6f}"
    )


    print(
        f"recon      : "
        f"{val['recon']:.6f}"
    )


    print(
        f"cross      : "
        f"{val['cross']:.6f}"
    )


    print(
        f"var        : "
        f"{val['var']:.6f}"
    )


    print()


    print(
        f"grid acc   : "
        f"{val['grid_acc']:.3f}"
    )


    print()


    print(
        f"World Δ@4  : "
        f"{val['delta_4']:.6f}"
    )


    print(
        f"World Δ@8  : "
        f"{val['delta_8']:.6f}"
    )


    print(
        f"World Δ@12 : "
        f"{val['delta_12']:.6f}"
    )


    # ========================================================
    # Checkpoint
    # ========================================================

    checkpoint = {

        "epoch":
            epoch,

        "global_step":
            global_step,

        "factorizer":
            factorizer.state_dict(),

        "recomposer":
            recomposer.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "config": {

            "phases":
                PHASES,

            "world_dim":
                WORLD_DIM,

            "grid_dim":
                GRID_DIM,

            "image_size":
                IMAGE_SIZE,

            "patch_size":
                PATCH_SIZE,

            "tubelet_size":
                TUBELET_SIZE,

            "num_frames":
                NUM_FRAMES,

            "num_train_clips":
                NUM_TRAIN_CLIPS,

            "num_val_clips":
                NUM_VAL_CLIPS,
        },

        "validation":
            val,
    }


    # --------------------------------------------------------
    # Save latest
    # --------------------------------------------------------

    last_path = (
        OUTPUT_DIR
        / "last.pt"
    )


    torch.save(

        checkpoint,

        last_path,
    )


    # --------------------------------------------------------
    # Save best
    # --------------------------------------------------------

    if (
        val["total"]
        < best_val
    ):

        best_val = (
            val["total"]
        )


        best_path = (
            OUTPUT_DIR
            / "best.pt"
        )


        torch.save(

            checkpoint,

            best_path,
        )


        print()

        print(
            "Saved new best checkpoint:"
        )

        print(
            best_path
        )


# ============================================================
# 16. Finished
# ============================================================

print()

print("=" * 80)

print(
    "TRAINING FINISHED"
)

print("=" * 80)


print(
    "Best checkpoint:"
)


print(
    OUTPUT_DIR
    / "best.pt"
)