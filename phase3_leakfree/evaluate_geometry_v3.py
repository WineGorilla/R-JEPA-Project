from pathlib import Path
import random

import numpy as np
import tensorflow_datasets as tfds
import torch
import torch.nn.functional as F

from factorizer import WorldGridFactorizer
from common_leakfree import amp_context, prepare_past_future, extract_z

SEED = 42
ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VJEPA_REPO = ROOT / "vjepa2"
DATA_DIR = ROOT / "data/tfds"

PHASE1_CKPT = ROOT / "checkpoints/phase1/best.pt"
PHASE3_CKPT = ROOT / "checkpoints/phase3_leakfree_v3/best.pt"
BASELINE_CKPT = ROOT / "checkpoints/phase3_baseline_leakfree/best.pt"

NUM_CLIPS = 128
TOKENS_PER_CLIP = 256
PHASE_SHIFT = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_encoder(trainable_state=None):
    encoder, unused = torch.hub.load(
        str(VJEPA_REPO),
        "vjepa2_1_vit_base_384",
        source="local",
    )
    del unused

    encoder = encoder.to(DEVICE)

    if trainable_state is not None:
        missing, unexpected = encoder.load_state_dict(
            trainable_state,
            strict=False,
        )

        print(
            f"Loaded partial encoder state: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )

    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


def make_factorizer(checkpoint):
    world_dim = checkpoint["config"]["world_dim"]
    grid_dim = checkpoint["config"]["grid_dim"]

    model = WorldGridFactorizer(
        input_dim=768,
        hidden_dim=768,
        world_dim=world_dim,
        grid_dim=grid_dim,
        num_grid_classes=4,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["factorizer"]
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def flatten_tokens(x):
    return x.reshape(
        -1,
        x.shape[-1],
    )


def pool_representation(x):
    return x.float().mean(
        dim=(1, 2, 3)
    )


def sample_tokens(x, num_tokens):
    x = flatten_tokens(x.float())

    if x.shape[0] <= num_tokens:
        return x

    indices = torch.randperm(
        x.shape[0],
        device=x.device,
    )[:num_tokens]

    return x[indices]


def cosine_distance(a, b):
    a = F.normalize(
        a.float(),
        dim=-1,
    )

    b = F.normalize(
        b.float(),
        dim=-1,
    )

    return (
        1.0
        - (a * b).sum(dim=-1)
    ).mean().item()


def effective_rank(x):
    x = x.float()
    x = x - x.mean(
        dim=0,
        keepdim=True,
    )

    singular_values = torch.linalg.svdvals(x)

    eigenvalues = singular_values.square()
    total = eigenvalues.sum()

    if total <= 1e-12:
        return 0.0, 0.0, eigenvalues

    probabilities = (
        eigenvalues / total
    )

    entropy = -(
        probabilities
        * torch.log(
            probabilities + 1e-12
        )
    ).sum()

    rank = torch.exp(entropy).item()

    max_rank = min(
        x.shape[0] - 1,
        x.shape[1],
    )

    normalized = (
        rank / max_rank
        if max_rank > 0
        else 0.0
    )

    return rank, normalized, eigenvalues


def explained_variance(eigenvalues):
    total = eigenvalues.sum()

    if total <= 1e-12:
        return 0.0, 0.0, 0.0

    ratios = (
        eigenvalues / total
    )

    top1 = ratios[:1].sum().item()
    top5 = ratios[:5].sum().item()
    top10 = ratios[:10].sum().item()

    return top1, top5, top10


def feature_std(x):
    return x.float().std(
        dim=0,
        unbiased=False,
    ).mean().item()


def mean_pairwise_cosine_distance(x):
    x = F.normalize(
        x.float(),
        dim=-1,
    )

    similarity = x @ x.T
    distance = 1.0 - similarity

    n = distance.shape[0]

    mask = torch.triu(
        torch.ones(
            n,
            n,
            dtype=torch.bool,
            device=distance.device,
        ),
        diagonal=1,
    )

    values = distance[mask]

    if values.numel() == 0:
        return 0.0

    return values.mean().item()


def calculate_metrics(global_features, token_features):
    global_features = torch.cat(
        global_features,
        dim=0,
    )

    token_features = torch.cat(
        token_features,
        dim=0,
    )

    global_rank, global_norm_rank, global_eigs = effective_rank(
        global_features
    )

    token_rank, token_norm_rank, token_eigs = effective_rank(
        token_features
    )

    global_top1, global_top5, global_top10 = explained_variance(
        global_eigs
    )

    token_top1, token_top5, token_top10 = explained_variance(
        token_eigs
    )

    return {
        "global_rank": global_rank,
        "global_norm_rank": global_norm_rank,
        "global_std": feature_std(global_features),
        "global_pairwise": mean_pairwise_cosine_distance(global_features),
        "global_top1": global_top1,
        "global_top5": global_top5,
        "global_top10": global_top10,
        "token_rank": token_rank,
        "token_norm_rank": token_norm_rank,
        "token_std": feature_std(token_features),
        "token_pairwise": mean_pairwise_cosine_distance(token_features),
        "token_top1": token_top1,
        "token_top5": token_top5,
        "token_top10": token_top10,
    }


print("=" * 100)
print("Loading checkpoints")
print("=" * 100)

phase1_ckpt = torch.load(
    PHASE1_CKPT,
    map_location=DEVICE,
)

phase3_ckpt = torch.load(
    PHASE3_CKPT,
    map_location=DEVICE,
)

baseline_ckpt = torch.load(
    BASELINE_CKPT,
    map_location=DEVICE,
)

print("Loading pretrained V-JEPA...")
pretrained_encoder = load_encoder()

print("Loading Phase-3 encoder...")
phase3_encoder = load_encoder(
    phase3_ckpt["encoder_trainable"]
)

print("Loading baseline encoder...")
baseline_encoder = load_encoder(
    baseline_ckpt["encoder_trainable"]
)

print("Loading Phase-1 factorizer...")
phase1_factorizer = make_factorizer(
    phase1_ckpt
)

print("Loading Phase-3 factorizer...")
phase3_factorizer = make_factorizer(
    phase3_ckpt
)

builder = tfds.builder(
    "movi_c/256x256",
    data_dir=str(DATA_DIR),
)

dataset = builder.as_dataset(
    split=f"test[:{NUM_CLIPS}]",
    shuffle_files=False,
)

names = [
    "Pretrained Raw",
    "Phase1 World",
    "Phase3 World",
    "Baseline Raw",
]

representations = {
    name: {
        "global": [],
        "tokens": [],
        "phase_delta": [],
        "same_video_global": [],
    }
    for name in names
}

print()
print("=" * 100)
print("Extracting representations")
print("=" * 100)

with torch.no_grad():
    for clip_idx, sample in enumerate(dataset):
        past, _ = prepare_past_future(sample)
        past = past.to(DEVICE)

        with amp_context(DEVICE):
            pretrained_z0 = extract_z(
                pretrained_encoder,
                past,
                0,
            )

            pretrained_z8 = extract_z(
                pretrained_encoder,
                past,
                PHASE_SHIFT,
            )

            phase1_w0, _, _ = phase1_factorizer(
                pretrained_z0
            )

            phase1_w8, _, _ = phase1_factorizer(
                pretrained_z8
            )

            phase3_z0 = extract_z(
                phase3_encoder,
                past,
                0,
            )

            phase3_z8 = extract_z(
                phase3_encoder,
                past,
                PHASE_SHIFT,
            )

            phase3_w0, _, _ = phase3_factorizer(
                phase3_z0
            )

            phase3_w8, _, _ = phase3_factorizer(
                phase3_z8
            )

            baseline_z0 = extract_z(
                baseline_encoder,
                past,
                0,
            )

            baseline_z8 = extract_z(
                baseline_encoder,
                past,
                PHASE_SHIFT,
            )

        current = {
            "Pretrained Raw": (
                pretrained_z0,
                pretrained_z8,
            ),
            "Phase1 World": (
                phase1_w0,
                phase1_w8,
            ),
            "Phase3 World": (
                phase3_w0,
                phase3_w8,
            ),
            "Baseline Raw": (
                baseline_z0,
                baseline_z8,
            ),
        }

        for name, (rep0, rep8) in current.items():
            pooled0 = pool_representation(
                rep0
            )

            pooled8 = pool_representation(
                rep8
            )

            tokens = sample_tokens(
                rep0,
                TOKENS_PER_CLIP,
            )

            representations[name]["global"].append(
                pooled0.float().cpu()
            )

            representations[name]["tokens"].append(
                tokens.float().cpu()
            )

            token_delta = cosine_distance(
                rep0.reshape(-1, rep0.shape[-1]),
                rep8.reshape(-1, rep8.shape[-1]),
            )

            global_delta = cosine_distance(
                pooled0,
                pooled8,
            )

            representations[name]["phase_delta"].append(
                token_delta
            )

            representations[name]["same_video_global"].append(
                global_delta
            )

        if clip_idx % 8 == 0:
            print(
                f"Finished clip {clip_idx}/{NUM_CLIPS}"
            )

print()
print("=" * 100)
print("Calculating geometry")
print("=" * 100)

all_metrics = {}

for name in names:
    metrics = calculate_metrics(
        representations[name]["global"],
        representations[name]["tokens"],
    )

    metrics["phase_delta"] = float(
        np.mean(
            representations[name]["phase_delta"]
        )
    )

    metrics["same_video_global"] = float(
        np.mean(
            representations[name]["same_video_global"]
        )
    )

    metrics["separation_ratio"] = (
        metrics["global_pairwise"]
        / max(
            metrics["same_video_global"],
            1e-8,
        )
    )

    all_metrics[name] = metrics

print()
print("=" * 135)
print("GLOBAL REPRESENTATION GEOMETRY")
print("=" * 135)

print(
    f"{'Representation':<20} | "
    f"{'Eff Rank':>9} | "
    f"{'Norm Rank':>9} | "
    f"{'Feat Std':>9} | "
    f"{'Pair Dist':>9} | "
    f"{'Top1 Var':>9} | "
    f"{'Top5 Var':>9} | "
    f"{'Top10 Var':>10}"
)

print("-" * 135)

for name in names:
    m = all_metrics[name]

    print(
        f"{name:<20} | "
        f"{m['global_rank']:>9.2f} | "
        f"{m['global_norm_rank'] * 100:>8.2f}% | "
        f"{m['global_std']:>9.5f} | "
        f"{m['global_pairwise']:>9.5f} | "
        f"{m['global_top1'] * 100:>8.2f}% | "
        f"{m['global_top5'] * 100:>8.2f}% | "
        f"{m['global_top10'] * 100:>9.2f}%"
    )

print()
print("=" * 135)
print("TOKEN-LEVEL REPRESENTATION GEOMETRY")
print("=" * 135)

print(
    f"{'Representation':<20} | "
    f"{'Eff Rank':>9} | "
    f"{'Norm Rank':>9} | "
    f"{'Feat Std':>9} | "
    f"{'Pair Dist':>9} | "
    f"{'Top1 Var':>9} | "
    f"{'Top5 Var':>9} | "
    f"{'Top10 Var':>10}"
)

print("-" * 135)

for name in names:
    m = all_metrics[name]

    print(
        f"{name:<20} | "
        f"{m['token_rank']:>9.2f} | "
        f"{m['token_norm_rank'] * 100:>8.2f}% | "
        f"{m['token_std']:>9.5f} | "
        f"{m['token_pairwise']:>9.5f} | "
        f"{m['token_top1'] * 100:>8.2f}% | "
        f"{m['token_top5'] * 100:>8.2f}% | "
        f"{m['token_top10'] * 100:>9.2f}%"
    )

print()
print("=" * 110)
print("CONTENT VS PHASE")
print("=" * 110)

print(
    f"{'Representation':<20} | "
    f"{'Token Phase Δ':>14} | "
    f"{'Global Phase Δ':>14} | "
    f"{'Inter-video Δ':>14} | "
    f"{'Separation':>12}"
)

print("-" * 110)

for name in names:
    m = all_metrics[name]

    print(
        f"{name:<20} | "
        f"{m['phase_delta']:>14.6f} | "
        f"{m['same_video_global']:>14.6f} | "
        f"{m['global_pairwise']:>14.6f} | "
        f"{m['separation_ratio']:>11.2f}x"
    )

print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)

raw_token_rank = all_metrics["Pretrained Raw"]["token_rank"]
phase1_token_rank = all_metrics["Phase1 World"]["token_rank"]
phase3_token_rank = all_metrics["Phase3 World"]["token_rank"]
baseline_token_rank = all_metrics["Baseline Raw"]["token_rank"]

print(
    f"Pretrained Raw token effective rank : {raw_token_rank:.2f}"
)
print(
    f"Phase-1 World token effective rank  : {phase1_token_rank:.2f}"
)
print(
    f"Phase-3 World token effective rank  : {phase3_token_rank:.2f}"
)
print(
    f"Baseline Raw token effective rank   : {baseline_token_rank:.2f}"
)

phase3_ratio = (
    phase3_token_rank
    / max(
        phase1_token_rank,
        1e-8,
    )
)

print()
print(
    f"Phase3 / Phase1 World token-rank ratio: "
    f"{phase3_ratio:.3f}"
)

if phase3_ratio >= 0.70:
    print(
        "PASS: Phase-3 retains most of the Phase-1 token-level effective rank."
    )
elif phase3_ratio >= 0.40:
    print(
        "CHECK: Phase-3 shows a noticeable reduction in token-level rank."
    )
else:
    print(
        "WARNING: Phase-3 World has a large token-level rank reduction."
    )

phase3_sep = all_metrics[
    "Phase3 World"
]["separation_ratio"]

if phase3_sep > 20:
    print(
        "PASS: Phase-3 World strongly separates content variation from phase variation."
    )
elif phase3_sep > 5:
    print(
        "PASS: Phase-3 World retains useful content/phase separation."
    )
else:
    print(
        "WARNING: Phase-3 content/phase separation is weak."
    )

print()
print("=" * 100)
print("DONE")
print("=" * 100)