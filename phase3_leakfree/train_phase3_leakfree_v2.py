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

from common_leakfree import (
    amp_context,
    prepare_past_future,
    extract_z,
    temporal_pair,
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

OUTPUT_DIR = ROOT / "checkpoints/phase3_leakfree_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_TRAIN_CLIPS = 2048
VAL_START = 168
NUM_VAL_CLIPS = 64

EPOCHS = 5
UNFREEZE_BLOCKS = 4

ENCODER_LR = 1e-5
FACTORIZER_LR = 5e-5
RECOMPOSER_LR = 5e-5
PREDICTOR_LR = 1e-4

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

LAMBDA_PRED = 0.75
LAMBDA_WORLD = 0.50
LAMBDA_GRID = 0.10
LAMBDA_RECON = 0.60
LAMBDA_CROSS = 0.60
LAMBDA_VAR = 0.075
LAMBDA_COV = 0.002

PHASE_TO_CLASS = {
    0: 0,
    4: 1,
    8: 2,
    12: 3,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def world_covariance_loss(world):
    x = world.reshape(-1, world.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)

    std = x.std(dim=0, unbiased=False).clamp_min(1e-4)
    x = x / std

    n = x.shape[0]

    if n <= 1:
        return x.new_tensor(0.0)

    cov = (x.T @ x) / (n - 1)

    diagonal = torch.diagonal(cov)
    off_diagonal = cov - torch.diag(diagonal)

    return off_diagonal.pow(2).sum() / x.shape[1]


print("=" * 80)
print("Loading V-JEPA")
print("=" * 80)

encoder, unused = torch.hub.load(
    str(VJEPA_REPO),
    "vjepa2_1_vit_base_384",
    source="local",
)

del unused

encoder = encoder.to(DEVICE)

freeze_encoder_except_last_blocks(
    encoder,
    UNFREEZE_BLOCKS,
)

phase1 = torch.load(
    PHASE1_CKPT,
    map_location=DEVICE,
)

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

factorizer.load_state_dict(
    phase1["factorizer"]
)

recomposer.load_state_dict(
    phase1["recomposer"]
)

world_predictor = TemporalPredictor(
    input_dim=WORLD_DIM,
    output_dim=WORLD_DIM,
    hidden_dim=384,
    num_layers=3,
    num_heads=6,
    num_horizons=3,
).to(DEVICE)

optimizer = AdamW(
    [
        {
            "params": trainable_parameters(encoder),
            "lr": ENCODER_LR,
        },
        {
            "params": factorizer.parameters(),
            "lr": FACTORIZER_LR,
        },
        {
            "params": recomposer.parameters(),
            "lr": RECOMPOSER_LR,
        },
        {
            "params": world_predictor.parameters(),
            "lr": PREDICTOR_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

all_trainable = (
    trainable_parameters(encoder)
    + list(factorizer.parameters())
    + list(recomposer.parameters())
    + list(world_predictor.parameters())
)

print("Device:", DEVICE)
print("World dim:", WORLD_DIM)
print("Grid dim:", GRID_DIM)

print(
    "Trainable params:",
    f"{sum(p.numel() for p in all_trainable) / 1e6:.2f}M",
)

print()
print("Loss weights")
print("pred :", LAMBDA_PRED)
print("world:", LAMBDA_WORLD)
print("grid :", LAMBDA_GRID)
print("recon:", LAMBDA_RECON)
print("cross:", LAMBDA_CROSS)
print("var  :", LAMBDA_VAR)
print("cov  :", LAMBDA_COV)

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
    factorizer.eval()
    world_predictor.eval()

    results = {
        0: {"pred": [], "copy": []},
        8: {"pred": [], "copy": []},
    }

    for sample in val_ds:
        past, future = prepare_past_future(sample)

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        for phase in [0, 8]:
            with amp_context(DEVICE):
                z_past = extract_z(
                    encoder,
                    past,
                    phase,
                )

                z_future = extract_z(
                    encoder,
                    future,
                    phase,
                )

                w_past, _, _ = factorizer(
                    z_past
                )

                w_future, _, _ = factorizer(
                    z_future
                )

                context, targets = temporal_pair(
                    w_past,
                    w_future,
                )

                predictions = world_predictor(
                    context
                )

                pred = multi_horizon_loss(
                    predictions,
                    targets,
                )["total"]

                copy = copy_last_loss(
                    context,
                    targets,
                )["total"]

            results[phase]["pred"].append(
                float(pred)
            )

            results[phase]["copy"].append(
                float(copy)
            )

    summary = {}

    for phase in [0, 8]:
        pred = float(
            np.mean(
                results[phase]["pred"]
            )
        )

        copy = float(
            np.mean(
                results[phase]["copy"]
            )
        )

        summary[phase] = {
            "pred": pred,
            "copy": copy,
            "gain": relative_gain(
                pred,
                copy,
            ),
        }

    return summary


print()
print("=" * 80)
print("Phase-3.1 leakage-free anti-collapse training")
print("=" * 80)

best_score = -float("inf")
global_step = 0

for epoch in range(1, EPOCHS + 1):
    set_partial_encoder_train_mode(
        encoder,
        UNFREEZE_BLOCKS,
    )

    factorizer.train()
    recomposer.train()
    world_predictor.train()

    totals = []
    preds = []
    worlds = []
    grids = []
    recons = []
    crosses = []
    vars_ = []
    covs = []

    print()
    print(f"Epoch {epoch}/{EPOCHS}")

    for clip_idx, sample in enumerate(train_ds):
        past, future = prepare_past_future(
            sample
        )

        past = past.to(DEVICE)
        future = future.to(DEVICE)

        phase_b = random.choice(
            [4, 8, 12]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with amp_context(DEVICE):
            zp0 = extract_z(
                encoder,
                past,
                0,
            )

            zpb = extract_z(
                encoder,
                past,
                phase_b,
            )

            zf0 = extract_z(
                encoder,
                future,
                0,
            )

            zfb = extract_z(
                encoder,
                future,
                phase_b,
            )

            wp0, gp0, lp0 = factorizer(zp0)
            wpb, gpb, lpb = factorizer(zpb)

            wf0, gf0, lf0 = factorizer(zf0)
            wfb, gfb, lfb = factorizer(zfb)

            world_loss = 0.5 * (
                world_consistency_loss(
                    wp0,
                    wpb,
                )
                + world_consistency_loss(
                    wf0,
                    wfb,
                )
            )

            label0 = torch.zeros(
                lp0.shape[0],
                dtype=torch.long,
                device=DEVICE,
            )

            labelb = torch.full(
                (lpb.shape[0],),
                PHASE_TO_CLASS[phase_b],
                dtype=torch.long,
                device=DEVICE,
            )

            grid_loss = 0.25 * (
                F.cross_entropy(
                    lp0,
                    label0,
                )
                + F.cross_entropy(
                    lpb,
                    labelb,
                )
                + F.cross_entropy(
                    lf0,
                    label0,
                )
                + F.cross_entropy(
                    lfb,
                    labelb,
                )
            )

            rp0 = recomposer(
                wp0,
                gp0,
            )

            rpb = recomposer(
                wpb,
                gpb,
            )

            rf0 = recomposer(
                wf0,
                gf0,
            )

            rfb = recomposer(
                wfb,
                gfb,
            )

            recon_loss = 0.25 * (
                representation_loss(
                    rp0,
                    zp0,
                )
                + representation_loss(
                    rpb,
                    zpb,
                )
                + representation_loss(
                    rf0,
                    zf0,
                )
                + representation_loss(
                    rfb,
                    zfb,
                )
            )

            cp01 = recomposer(
                wp0,
                gpb,
            )

            cp10 = recomposer(
                wpb,
                gp0,
            )

            cf01 = recomposer(
                wf0,
                gfb,
            )

            cf10 = recomposer(
                wfb,
                gf0,
            )

            cross_loss = 0.25 * (
                representation_loss(
                    cp01,
                    zpb,
                )
                + representation_loss(
                    cp10,
                    zp0,
                )
                + representation_loss(
                    cf01,
                    zfb,
                )
                + representation_loss(
                    cf10,
                    zf0,
                )
            )

            var_loss = 0.25 * (
                world_variance_loss(wp0)
                + world_variance_loss(wpb)
                + world_variance_loss(wf0)
                + world_variance_loss(wfb)
            )

            cov_loss = 0.25 * (
                world_covariance_loss(wp0)
                + world_covariance_loss(wpb)
                + world_covariance_loss(wf0)
                + world_covariance_loss(wfb)
            )

            context0, targets0 = temporal_pair(
                wp0,
                wf0,
            )

            contextb, targetsb = temporal_pair(
                wpb,
                wfb,
            )

            predictions0 = world_predictor(
                context0
            )

            predictionsb = world_predictor(
                contextb
            )

            pred_loss0 = multi_horizon_loss(
                predictions0,
                targets0,
            )["total"]

            pred_lossb = multi_horizon_loss(
                predictionsb,
                targetsb,
            )["total"]

            pred_loss = 0.5 * (
                pred_loss0
                + pred_lossb
            )

            total_loss = (
                LAMBDA_PRED * pred_loss
                + LAMBDA_WORLD * world_loss
                + LAMBDA_GRID * grid_loss
                + LAMBDA_RECON * recon_loss
                + LAMBDA_CROSS * cross_loss
                + LAMBDA_VAR * var_loss
                + LAMBDA_COV * cov_loss
            )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            all_trainable,
            GRAD_CLIP,
        )

        optimizer.step()

        global_step += 1

        totals.append(
            float(total_loss)
        )

        preds.append(
            float(pred_loss)
        )

        worlds.append(
            float(world_loss)
        )

        grids.append(
            float(grid_loss)
        )

        recons.append(
            float(recon_loss)
        )

        crosses.append(
            float(cross_loss)
        )

        vars_.append(
            float(var_loss)
        )

        covs.append(
            float(cov_loss)
        )

        if clip_idx % 25 == 0:
            print(
                f"clip={clip_idx:04d} "
                f"phase={phase_b:02d} "
                f"total={float(total_loss):.4f} "
                f"pred={float(pred_loss):.4f} "
                f"world={float(world_loss):.4f} "
                f"grid={float(grid_loss):.4f} "
                f"recon={float(recon_loss):.4f} "
                f"cross={float(cross_loss):.4f} "
                f"var={float(var_loss):.4f} "
                f"cov={float(cov_loss):.4f}"
            )

    print()
    print(f"Train total : {np.mean(totals):.6f}")
    print(f"Train pred  : {np.mean(preds):.6f}")
    print(f"Train world : {np.mean(worlds):.6f}")
    print(f"Train grid  : {np.mean(grids):.6f}")
    print(f"Train recon : {np.mean(recons):.6f}")
    print(f"Train cross : {np.mean(crosses):.6f}")
    print(f"Train var   : {np.mean(vars_):.6f}")
    print(f"Train cov   : {np.mean(covs):.6f}")

    validation = validate()

    print()
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
        "encoder_trainable": trainable_state_dict(
            encoder
        ),
        "factorizer": factorizer.state_dict(),
        "recomposer": recomposer.state_dict(),
        "world_predictor": world_predictor.state_dict(),
        "validation": validation,
        "config": {
            "world_dim": WORLD_DIM,
            "grid_dim": GRID_DIM,
            "unfreeze_blocks": UNFREEZE_BLOCKS,
            "protocol": "leakage_free_v2_anticollapse",
            "lambda_pred": LAMBDA_PRED,
            "lambda_world": LAMBDA_WORLD,
            "lambda_grid": LAMBDA_GRID,
            "lambda_recon": LAMBDA_RECON,
            "lambda_cross": LAMBDA_CROSS,
            "lambda_var": LAMBDA_VAR,
            "lambda_cov": LAMBDA_COV,
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

        print(
            "Saved new best Phase-3.1 checkpoint."
        )

print()
print(
    "Phase-3.1 anti-collapse training finished."
)