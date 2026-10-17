from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
QA_DIR = ROOT / "clevrer_qa"
CHECKPOINT_ROOT = QA_DIR / "checkpoints"
RESULT_DIR = QA_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(QA_DIR))

from train_qa import (
    CLEVRERQADataset,
    QAProbe,
    collate_fn,
    evaluate,
    print_metrics,
    load_json,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_representation(representation, batch_size):
    checkpoint_dir = CHECKPOINT_ROOT / representation
    checkpoint_path = checkpoint_dir / "best.pt"
    vocab_path = checkpoint_dir / "vocab.json"
    answer_vocab_path = checkpoint_dir / "answer_vocab.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing QA checkpoint: {checkpoint_path}"
        )

    with vocab_path.open("r") as f:
        vocab = json.load(f)

    with answer_vocab_path.open("r") as f:
        answer_vocab = json.load(f)

    ckpt = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    config = ckpt["config"]

    dataset = CLEVRERQADataset(
        "validation",
        representation,
        vocab,
        answer_vocab,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = QAProbe(
        video_dim=config["video_dim"],
        vocab_size=config["vocab_size"],
        num_answers=config["num_answers"],
    ).to(DEVICE)

    model.load_state_dict(
        ckpt["model"]
    )
    model.eval()

    metrics = evaluate(
        model,
        loader,
    )

    return metrics, ckpt


def fmt(x):
    if x is None or not np.isfinite(x):
        return "   n/a"
    return f"{x * 100:6.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    all_results = {}

    for representation in ["raw", "world"]:
        metrics, ckpt = load_representation(
            representation,
            args.batch_size,
        )
        all_results[representation] = {
            "metrics": metrics,
            "checkpoint_epoch": ckpt["epoch"],
        }

        print_metrics(
            metrics,
            f"{representation.upper()} FINAL VALIDATION",
        )

    raw = all_results["raw"]["metrics"]
    world = all_results["world"]["metrics"]

    columns = [
        ("Desc-Q", "descriptive_per_question"),
        ("Pred-O", "predictive_per_option"),
        ("Pred-Q", "predictive_per_question"),
        ("Expl-O", "explanatory_per_option"),
        ("Expl-Q", "explanatory_per_question"),
        ("CF-O", "counterfactual_per_option"),
        ("CF-Q", "counterfactual_per_question"),
        ("Overall-Q", "overall_per_question"),
    ]

    print()
    print("=" * 132)
    print("CLEVRER QA FINAL COMPARISON")
    print("=" * 132)

    header = (
        f"{'Representation':<16}"
        + "".join(
            f"{name:>13}"
            for name, _ in columns
        )
    )
    print(header)
    print("-" * 132)

    for name, metrics in [
        ("V-JEPA Raw", raw),
        ("R-JEPA World", world),
    ]:
        row = f"{name:<16}"

        for _, key in columns:
            row += f"{fmt(metrics.get(key)):>13}"

        print(row)

    print("-" * 132)

    row = f"{'World - Raw':<16}"
    for _, key in columns:
        a = raw.get(key, float("nan"))
        b = world.get(key, float("nan"))

        if np.isfinite(a) and np.isfinite(b):
            diff = (b - a) * 100.0
            row += f"{diff:+12.2f}pp"
        else:
            row += f"{'n/a':>13}"

    print(row)

    output_path = RESULT_DIR / "clevrer_qa_results.json"
    with output_path.open("w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
