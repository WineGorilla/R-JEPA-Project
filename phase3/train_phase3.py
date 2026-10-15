import random
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from factorizer import WorldGridFactorizer
from recomposer import WorldGridRecomposer
from temporal_predictor import TemporalPredictor, multi_horizon_loss
from common import (
    amp_context,
    prepare_video,
    extract_z,
    split_temporal,
    representation_loss,
    world_consistency_loss,
    world_variance_loss,
    copy_last_loss,
    relative_gain,
    freeze_encoder_except_last_blocks,
    set_partial_encoder_train_mode,
    trainable_parameters,
    trainable_state_dict,
)

SEED = 42
ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VJEPA_REPO = ROOT / "vjepa2"
DATA_DIR = ROOT / "data/tfds"
PHASE1_CKPT = ROOT / "checkpoints/phase1/best.pt"
PHASE2_CKPT = ROOT / "checkpoints/phase2/best.pt"
OUTPUT_DIR = ROOT / "checkpoints/phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TRAIN_CLIPS = 2048
VAL_START = 168
NUM_VAL_CLIPS = 64
EPOCHS = 5

TRAIN_PHASES = [0, 4, 8, 12]
PHASE_TO_CLASS = {0: 0, 4: 1, 8: 2, 12: 3}

UNFREEZE_BLOCKS = 4
ENCODER_LR = 1e-5
FACTORIZER_LR = 5e-5
RECOMPOSER_LR = 5e-5
PREDICTOR_LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

LAMBDA_PRED = 1.0
LAMBDA_WORLD = 0.25
LAMBDA_GRID = 0.10
LAMBDA_RECON = 0.50
LAMBDA_CROSS = 0.50
LAMBDA_VAR = 0.05

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("Loading V-JEPA")
print("=" * 80)

encoder, unused_predictor = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)
del unused_predictor
encoder = encoder.to(DEVICE)

freeze_encoder_except_last_blocks(encoder, UNFREEZE_BLOCKS)

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Trainable encoder blocks:", UNFREEZE_BLOCKS)

phase1 = torch.load(PHASE1_CKPT, map_location=DEVICE)
phase2 = torch.load(PHASE2_CKPT, map_location=DEVICE)

WORLD_DIM = phase1["config"]["world_dim"]
GRID_DIM = phase1["config"]["grid_dim"]

factorizer = WorldGridFactorizer(
    input_dim=768,
    hidden_dim=768,
    world_dim=WORLD_DIM,
    grid_dim=GRID_DIM,
    num_grid_classes=4,
).to(DEVICE)

recomposer = WorldGridRecomposer(
    world_dim=WORLD_DIM,
    grid_dim=GRID_DIM,
    hidden_dim=768,
    output_dim=768,
).to(DEVICE)

factorizer.load_state_dict(phase1["factorizer"])
recomposer.load_state_dict(phase1["recomposer"])

world_predictor = TemporalPredictor(
    input_dim=WORLD_DIM,
    output_dim=WORLD_DIM,
    hidden_dim=phase2["config"]["hidden_dim"],
    num_layers=phase2["config"]["layers"],
    num_heads=phase2["config"]["heads"],
    num_horizons=len(phase2["config"]["target_indices"]),
).to(DEVICE)

world_predictor.load_state_dict(phase2["world_predictor"])

optimizer = AdamW(
    [
        {
            "params": trainable_parameters(encoder),
            "lr": ENCODER_LR,
            "name": "encoder",
        },
        {
            "params": factorizer.parameters(),
            "lr": FACTORIZER_LR,
            "name": "factorizer",
        },
        {
            "params": recomposer.parameters(),
            "lr": RECOMPOSER_LR,
            "name": "recomposer",
        },
        {
            "params": world_predictor.parameters(),
            "lr": PREDICTOR_LR,
            "name": "predictor",
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

all_trainable_params = (
    trainable_parameters(encoder)
    + list(factorizer.parameters())
    + list(recomposer.parameters())
    + list(world_predictor.parameters())
)

print("World dim:", WORLD_DIM)
print("Grid dim:", GRID_DIM)
print(
    "Trainable parameters:",
    f"{sum(p.numel() for p in all_trainable_params) / 1e6:.2f}M",
)

builder = tfds.builder(
    "movi_c/256x256",
    data_dir=str(DATA_DIR),
)

train_ds = builder.as_dataset(
    split=f"train[:{NUM_TRAIN_CLIPS}]",
    shuffle_files=True,
)

val_ds = builder.as_dataset(
    split=f"validation[{VAL_START}:{VAL_START + NUM_VAL_CLIPS}]",
    shuffle_files=False,
)

print("Train clips:", NUM_TRAIN_CLIPS)
print("Validation clips:", NUM_VAL_CLIPS)


@torch.no_grad()
def validate():
    encoder.eval()
    factorizer.eval()
    recomposer.eval()
    world_predictor.eval()

    results = {
        0: {"pred": [], "copy": []},
        8: {"pred": [], "copy": []},
    }

    for sample in val_ds:
        video = prepare_video(sample).to(DEVICE)

        for phase in [0, 8]:
            with amp_context(DEVICE):
                z = extract_z(encoder, video, phase)
                world, _, _ = factorizer(z)

                context, targets = split_temporal(world)
                predictions = world_predictor(context)

                pred_loss = multi_horizon_loss(
                    predictions,
                    targets,
                )["total"]

                copy_loss = copy_last_loss(
                    context,
                    targets,
                )["total"]

            results[phase]["pred"].append(float(pred_loss))
            results[phase]["copy"].append(float(copy_loss))

    summary = {}

    for phase in [0, 8]:
        pred = float(np.mean(results[phase]["pred"]))
        copy = float(np.mean(results[phase]["copy"]))
        summary[phase] = {
            "pred": pred,
            "copy": copy,
            "gain": relative_gain(pred, copy),
        }

    return summary


print()
print("=" * 80)
print("Starting Phase-3 joint fine-tuning")
print("=" * 80)

best_score = -float("inf")
global_step = 0

for epoch in range(1, EPOCHS + 1):
    set_partial_encoder_train_mode(encoder, UNFREEZE_BLOCKS)
    factorizer.train()
    recomposer.train()
    world_predictor.train()

    epoch_total = []
    epoch_pred = []
    epoch_world = []
    epoch_grid = []
    epoch_recon = []
    epoch_cross = []

    print()
    print(f"Epoch {epoch}/{EPOCHS}")

    for clip_idx, sample in enumerate(train_ds):
        video = prepare_video(sample).to(DEVICE)

        phase_b = random.choice([4, 8, 12])
        class_a = PHASE_TO_CLASS[0]
        class_b = PHASE_TO_CLASS[phase_b]

        optimizer.zero_grad(set_to_none=True)

        with amp_context(DEVICE):
            z_a = extract_z(encoder, video, 0)
            z_b = extract_z(encoder, video, phase_b)

            w_a, g_a, logits_a = factorizer(z_a)
            w_b, g_b, logits_b = factorizer(z_b)

            world_loss = world_consistency_loss(w_a, w_b)

            labels_a = torch.full(
                (logits_a.shape[0],),
                class_a,
                dtype=torch.long,
                device=DEVICE,
            )

            labels_b = torch.full(
                (logits_b.shape[0],),
                class_b,
                dtype=torch.long,
                device=DEVICE,
            )

            grid_loss = 0.5 * (
                F.cross_entropy(logits_a, labels_a)
                + F.cross_entropy(logits_b, labels_b)
            )

            zhat_a = recomposer(w_a, g_a)
            zhat_b = recomposer(w_b, g_b)

            recon_loss = 0.5 * (
                representation_loss(zhat_a, z_a)
                + representation_loss(zhat_b, z_b)
            )

            cross_ab = recomposer(w_a, g_b)
            cross_ba = recomposer(w_b, g_a)

            cross_loss = 0.5 * (
                representation_loss(cross_ab, z_b)
                + representation_loss(cross_ba, z_a)
            )

            var_loss = 0.5 * (
                world_variance_loss(w_a)
                + world_variance_loss(w_b)
            )

            context_a, targets_a = split_temporal(w_a)
            context_b, targets_b = split_temporal(w_b)

            predictions_a = world_predictor(context_a)
            predictions_b = world_predictor(context_b)

            pred_loss_a = multi_horizon_loss(
                predictions_a,
                targets_a,
            )["total"]

            pred_loss_b = multi_horizon_loss(
                predictions_b,
                targets_b,
            )["total"]

            pred_loss = 0.5 * (
                pred_loss_a
                + pred_loss_b
            )

            total_loss = (
                LAMBDA_PRED * pred_loss
                + LAMBDA_WORLD * world_loss
                + LAMBDA_GRID * grid_loss
                + LAMBDA_RECON * recon_loss
                + LAMBDA_CROSS * cross_loss
                + LAMBDA_VAR * var_loss
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            all_trainable_params,
            GRAD_CLIP,
        )

        optimizer.step()
        global_step += 1

        epoch_total.append(float(total_loss))
        epoch_pred.append(float(pred_loss))
        epoch_world.append(float(world_loss))
        epoch_grid.append(float(grid_loss))
        epoch_recon.append(float(recon_loss))
        epoch_cross.append(float(cross_loss))

        if clip_idx % 25 == 0:
            print(
                f"clip={clip_idx:04d} phase={phase_b:02d} "
                f"total={float(total_loss):.4f} "
                f"pred={float(pred_loss):.4f} "
                f"world={float(world_loss):.4f} "
                f"grid={float(grid_loss):.4f} "
                f"recon={float(recon_loss):.4f} "
                f"cross={float(cross_loss):.4f}"
            )

    print()
    print(f"Train total : {np.mean(epoch_total):.6f}")
    print(f"Train pred  : {np.mean(epoch_pred):.6f}")
    print(f"Train world : {np.mean(epoch_world):.6f}")
    print(f"Train grid  : {np.mean(epoch_grid):.6f}")
    print(f"Train recon : {np.mean(epoch_recon):.6f}")
    print(f"Train cross : {np.mean(epoch_cross):.6f}")

    validation = validate()

    print()
    print("Validation")
    for phase in [0, 8]:
        r = validation[phase]
        print(
            f"{phase}px | pred={r['pred']:.6f} "
            f"copy={r['copy']:.6f} "
            f"gain={r['gain'] * 100:.2f}%"
        )

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder_trainable": trainable_state_dict(encoder),
        "factorizer": factorizer.state_dict(),
        "recomposer": recomposer.state_dict(),
        "world_predictor": world_predictor.state_dict(),
        "validation": validation,
        "config": {
            "world_dim": WORLD_DIM,
            "grid_dim": GRID_DIM,
            "unfreeze_blocks": UNFREEZE_BLOCKS,
            "train_phases": TRAIN_PHASES,
            "encoder_lr": ENCODER_LR,
            "factorizer_lr": FACTORIZER_LR,
            "recomposer_lr": RECOMPOSER_LR,
            "predictor_lr": PREDICTOR_LR,
        },
    }

    torch.save(
        checkpoint,
        OUTPUT_DIR / "last.pt",
    )

    score = 0.5 * (
        validation[0]["gain"]
        + validation[8]["gain"]
    )

    if score > best_score:
        best_score = score
        torch.save(
            checkpoint,
            OUTPUT_DIR / "best.pt",
        )
        print("Saved new best Phase-3 checkpoint.")

print()
print("Phase-3 training finished.")