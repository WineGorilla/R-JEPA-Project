import random
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
from torch.optim import AdamW

from temporal_predictor import TemporalPredictor, multi_horizon_loss
from common import (
    amp_context,
    prepare_video,
    extract_z,
    split_temporal,
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
PHASE2_CKPT = ROOT / "checkpoints/phase2/best.pt"
OUTPUT_DIR = ROOT / "checkpoints/phase3_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TRAIN_CLIPS = 2048
VAL_START = 168
NUM_VAL_CLIPS = 64
EPOCHS = 5

UNFREEZE_BLOCKS = 4
ENCODER_LR = 1e-5
PREDICTOR_LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 80)
print("Loading matched V-JEPA baseline")
print("=" * 80)

encoder, unused_predictor = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)
del unused_predictor
encoder = encoder.to(DEVICE)

freeze_encoder_except_last_blocks(
    encoder,
    UNFREEZE_BLOCKS,
)

phase2 = torch.load(
    PHASE2_CKPT,
    map_location=DEVICE,
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
    phase2["raw_predictor"]
)

optimizer = AdamW(
    [
        {
            "params": trainable_parameters(encoder),
            "lr": ENCODER_LR,
        },
        {
            "params": raw_predictor.parameters(),
            "lr": PREDICTOR_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

all_trainable_params = (
    trainable_parameters(encoder)
    + list(raw_predictor.parameters())
)

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


@torch.no_grad()
def validate():
    encoder.eval()
    raw_predictor.eval()

    results = {
        0: {"pred": [], "copy": []},
        8: {"pred": [], "copy": []},
    }

    for sample in val_ds:
        video = prepare_video(sample).to(DEVICE)

        for phase in [0, 8]:
            with amp_context(DEVICE):
                z = extract_z(
                    encoder,
                    video,
                    phase,
                )

                context, targets = split_temporal(z)
                predictions = raw_predictor(context)

                pred = multi_horizon_loss(
                    predictions,
                    targets,
                )["total"]

                copy = copy_last_loss(
                    context,
                    targets,
                )["total"]

            results[phase]["pred"].append(float(pred))
            results[phase]["copy"].append(float(copy))

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
print("Starting matched baseline fine-tuning")
print("=" * 80)

best_score = -float("inf")
global_step = 0

for epoch in range(1, EPOCHS + 1):
    set_partial_encoder_train_mode(
        encoder,
        UNFREEZE_BLOCKS,
    )

    raw_predictor.train()
    epoch_losses = []

    print()
    print(f"Epoch {epoch}/{EPOCHS}")

    for clip_idx, sample in enumerate(train_ds):
        video = prepare_video(sample).to(DEVICE)
        phase_b = random.choice([4, 8, 12])

        optimizer.zero_grad(set_to_none=True)

        with amp_context(DEVICE):
            z_a = extract_z(
                encoder,
                video,
                0,
            )

            z_b = extract_z(
                encoder,
                video,
                phase_b,
            )

            context_a, targets_a = split_temporal(z_a)
            context_b, targets_b = split_temporal(z_b)

            predictions_a = raw_predictor(context_a)
            predictions_b = raw_predictor(context_b)

            loss_a = multi_horizon_loss(
                predictions_a,
                targets_a,
            )["total"]

            loss_b = multi_horizon_loss(
                predictions_b,
                targets_b,
            )["total"]

            loss = 0.5 * (
                loss_a
                + loss_b
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            all_trainable_params,
            GRAD_CLIP,
        )

        optimizer.step()
        global_step += 1
        epoch_losses.append(float(loss))

        if clip_idx % 25 == 0:
            print(
                f"clip={clip_idx:04d} "
                f"phase={phase_b:02d} "
                f"loss={float(loss):.6f}"
            )

    print(
        f"Train loss: {np.mean(epoch_losses):.6f}"
    )

    validation = validate()

    print("Validation")
    for phase in [0, 8]:
        r = validation[phase]

        print(
            f"{phase}px | "
            f"pred={r['pred']:.6f} "
            f"copy={r['copy']:.6f} "
            f"gain={r['gain'] * 100:.2f}%"
        )

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder_trainable": trainable_state_dict(encoder),
        "raw_predictor": raw_predictor.state_dict(),
        "validation": validation,
        "config": {
            "unfreeze_blocks": UNFREEZE_BLOCKS,
            "encoder_lr": ENCODER_LR,
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

        print("Saved new best baseline checkpoint.")

print()
print("Baseline training finished.")