from pathlib import Path
import argparse
import json
import math
import random
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/data/mi2-interns/ruiyu/R-JEPA")
DATA_ROOT = ROOT / "data/clevrer"
CACHE_ROOT = ROOT / "clevrer_qa/cache"
OUTPUT_ROOT = ROOT / "clevrer_qa/checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
EPOCHS = 20
BATCH_SIZE = 128
LR = 3e-4
WEIGHT_DECAY = 1e-4
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
FF_DIM = 1024
DROPOUT = 0.1
MAX_TEXT_LEN = 64
MAX_VIDEO_STEPS = 32

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sep>"]
PAD_ID = 0
UNK_ID = 1
SEP_ID = 2


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]",
        text.lower(),
    )


def load_json(split):
    path = DATA_ROOT / "questions" / f"{split}.json"
    with path.open("r") as f:
        return json.load(f)


def build_vocab(train_data):
    words = set()

    for item in train_data:
        for q in item["questions"]:
            words.update(tokenize(q["question"]))
            for choice in q.get("choices", []):
                words.update(tokenize(choice["choice"]))

    vocab = {
        token: idx
        for idx, token in enumerate(
            SPECIAL_TOKENS + sorted(words)
        )
    }
    return vocab


def build_answer_vocab(train_data):
    answers = sorted(
        {
            q["answer"]
            for item in train_data
            for q in item["questions"]
            if q["question_type"] == "descriptive"
        }
    )
    return {
        answer: idx
        for idx, answer in enumerate(answers)
    }


def encode_text(question, vocab, choice=None):
    tokens = tokenize(question)

    if choice is not None:
        tokens += ["<sep>"] + tokenize(choice)

    ids = [
        vocab.get(token, UNK_ID)
        for token in tokens[:MAX_TEXT_LEN]
    ]

    if not ids:
        ids = [UNK_ID]

    return ids


class CLEVRERQADataset(Dataset):
    def __init__(
        self,
        split,
        representation,
        vocab,
        answer_vocab,
    ):
        self.split = split
        self.representation = representation
        self.vocab = vocab
        self.answer_vocab = answer_vocab

        rep_dir = CACHE_ROOT / representation
        feature_path = rep_dir / f"{split}_features.npy"
        index_path = rep_dir / f"{split}_index.json"

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Missing features: {feature_path}\n"
                f"Run extract_features.py first."
            )

        self.features = np.load(
            feature_path,
            mmap_mode="r",
        )

        with index_path.open("r") as f:
            self.video_index = json.load(f)

        data = load_json(split)
        self.examples = []

        for item in data:
            video_name = item["video_filename"]

            if video_name not in self.video_index:
                continue

            video_idx = self.video_index[video_name]

            for q in item["questions"]:
                qtype = q["question_type"]
                qid = (
                    f"{item['scene_index']}:"
                    f"{q['question_id']}"
                )

                if qtype == "descriptive":
                    answer = q["answer"]

                    if answer not in answer_vocab:
                        continue

                    self.examples.append(
                        {
                            "video_idx": video_idx,
                            "text_ids": encode_text(
                                q["question"],
                                vocab,
                            ),
                            "task": 0,
                            "target": answer_vocab[answer],
                            "qtype": qtype,
                            "qid": qid,
                            "choice_id": -1,
                        }
                    )
                else:
                    for choice in q.get("choices", []):
                        target = (
                            1
                            if choice["answer"] == "correct"
                            else 0
                        )

                        self.examples.append(
                            {
                                "video_idx": video_idx,
                                "text_ids": encode_text(
                                    q["question"],
                                    vocab,
                                    choice["choice"],
                                ),
                                "task": 1,
                                "target": target,
                                "qtype": qtype,
                                "qid": qid,
                                "choice_id": choice["choice_id"],
                            }
                        )

        print(
            f"{split}: {len(self.examples):,} flattened QA examples "
            f"from {len(self.video_index):,} cached videos"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        video = np.asarray(
            self.features[ex["video_idx"]],
            dtype=np.float32,
        )

        return {
            "video": torch.from_numpy(video.copy()),
            "text_ids": torch.tensor(
                ex["text_ids"],
                dtype=torch.long,
            ),
            "task": ex["task"],
            "target": ex["target"],
            "qtype": ex["qtype"],
            "qid": ex["qid"],
            "choice_id": ex["choice_id"],
        }


def collate_fn(batch):
    max_len = min(
        MAX_TEXT_LEN,
        max(len(x["text_ids"]) for x in batch),
    )

    text = torch.full(
        (len(batch), max_len),
        PAD_ID,
        dtype=torch.long,
    )

    text_mask = torch.zeros(
        (len(batch), max_len),
        dtype=torch.bool,
    )

    for i, item in enumerate(batch):
        ids = item["text_ids"][:max_len]
        text[i, :len(ids)] = ids
        text_mask[i, :len(ids)] = True

    return {
        "video": torch.stack(
            [x["video"] for x in batch]
        ),
        "text_ids": text,
        "text_mask": text_mask,
        "task": torch.tensor(
            [x["task"] for x in batch],
            dtype=torch.long,
        ),
        "target": torch.tensor(
            [x["target"] for x in batch],
            dtype=torch.long,
        ),
        "qtype": [x["qtype"] for x in batch],
        "qid": [x["qid"] for x in batch],
        "choice_id": [x["choice_id"] for x in batch],
    }


class QAProbe(nn.Module):
    def __init__(
        self,
        video_dim,
        vocab_size,
        num_answers,
    ):
        super().__init__()

        self.video_norm = nn.LayerNorm(video_dim)
        self.video_proj = nn.Linear(
            video_dim,
            D_MODEL,
        )

        self.text_embed = nn.Embedding(
            vocab_size,
            D_MODEL,
            padding_idx=PAD_ID,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, D_MODEL)
        )

        self.sep_token = nn.Parameter(
            torch.zeros(1, 1, D_MODEL)
        )

        self.video_pos = nn.Parameter(
            torch.zeros(
                1,
                MAX_VIDEO_STEPS,
                D_MODEL,
            )
        )

        self.text_pos = nn.Parameter(
            torch.zeros(
                1,
                MAX_TEXT_LEN,
                D_MODEL,
            )
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS,
        )

        self.final_norm = nn.LayerNorm(D_MODEL)

        self.desc_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, num_answers),
        )

        self.binary_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, 2),
        )

        nn.init.normal_(
            self.cls_token,
            std=0.02,
        )
        nn.init.normal_(
            self.sep_token,
            std=0.02,
        )
        nn.init.normal_(
            self.video_pos,
            std=0.02,
        )
        nn.init.normal_(
            self.text_pos,
            std=0.02,
        )

    def forward(
        self,
        video,
        text_ids,
        text_mask,
    ):
        b, t, _ = video.shape

        if t > MAX_VIDEO_STEPS:
            raise RuntimeError(
                f"Video has {t} steps; "
                f"MAX_VIDEO_STEPS={MAX_VIDEO_STEPS}"
            )

        video = self.video_proj(
            self.video_norm(video)
        )
        video = video + self.video_pos[:, :t]

        text = self.text_embed(text_ids)
        text = text + self.text_pos[:, :text.shape[1]]

        cls = self.cls_token.expand(
            b,
            -1,
            -1,
        )
        sep = self.sep_token.expand(
            b,
            -1,
            -1,
        )

        x = torch.cat(
            [cls, video, sep, text],
            dim=1,
        )

        prefix_mask = torch.ones(
            b,
            1 + t + 1,
            dtype=torch.bool,
            device=video.device,
        )

        valid_mask = torch.cat(
            [
                prefix_mask,
                text_mask,
            ],
            dim=1,
        )

        x = self.transformer(
            x,
            src_key_padding_mask=~valid_mask,
        )

        pooled = self.final_norm(
            x[:, 0]
        )

        return (
            self.desc_head(pooled),
            self.binary_head(pooled),
        )


def compute_loss(
    desc_logits,
    binary_logits,
    task,
    target,
):
    losses = []

    desc_mask = task == 0
    binary_mask = task == 1

    if desc_mask.any():
        losses.append(
            nn.functional.cross_entropy(
                desc_logits[desc_mask],
                target[desc_mask],
            )
        )

    if binary_mask.any():
        losses.append(
            nn.functional.cross_entropy(
                binary_logits[binary_mask],
                target[binary_mask],
            )
        )

    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    desc_correct = 0
    desc_total = 0

    option_correct = {
        "explanatory": 0,
        "predictive": 0,
        "counterfactual": 0,
    }
    option_total = {
        k: 0 for k in option_correct
    }

    grouped = {}

    for batch in loader:
        video = batch["video"].to(DEVICE)
        text_ids = batch["text_ids"].to(DEVICE)
        text_mask = batch["text_mask"].to(DEVICE)
        task = batch["task"].to(DEVICE)
        target = batch["target"].to(DEVICE)

        desc_logits, binary_logits = model(
            video,
            text_ids,
            text_mask,
        )

        desc_pred = desc_logits.argmax(dim=-1)
        binary_pred = binary_logits.argmax(dim=-1)

        for i in range(len(batch["qid"])):
            qtype = batch["qtype"][i]
            qid = batch["qid"][i]
            gold = int(target[i].item())

            if int(task[i].item()) == 0:
                pred = int(desc_pred[i].item())
                desc_correct += int(pred == gold)
                desc_total += 1
                grouped[qid] = {
                    "type": "descriptive",
                    "correct": pred == gold,
                }
            else:
                pred = int(binary_pred[i].item())

                if qtype not in option_correct:
                    option_correct[qtype] = 0
                    option_total[qtype] = 0

                option_correct[qtype] += int(pred == gold)
                option_total[qtype] += 1

                if qid not in grouped:
                    grouped[qid] = {
                        "type": qtype,
                        "choices": [],
                    }

                grouped[qid]["choices"].append(
                    pred == gold
                )

    per_question = {
        "descriptive": [],
        "explanatory": [],
        "predictive": [],
        "counterfactual": [],
    }

    for item in grouped.values():
        qtype = item["type"]

        if qtype == "descriptive":
            correct = item["correct"]
        else:
            correct = all(item["choices"])

        if qtype not in per_question:
            per_question[qtype] = []

        per_question[qtype].append(
            float(correct)
        )

    metrics = {
        "descriptive_per_question": (
            desc_correct / desc_total
            if desc_total else float("nan")
        )
    }

    for qtype in [
        "predictive",
        "explanatory",
        "counterfactual",
    ]:
        metrics[f"{qtype}_per_option"] = (
            option_correct.get(qtype, 0)
            / max(option_total.get(qtype, 0), 1)
        )

        values = per_question.get(qtype, [])
        metrics[f"{qtype}_per_question"] = (
            float(np.mean(values))
            if values else float("nan")
        )

    all_questions = [
        x
        for values in per_question.values()
        for x in values
    ]

    metrics["overall_per_question"] = (
        float(np.mean(all_questions))
        if all_questions else float("nan")
    )

    return metrics


def print_metrics(metrics, title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)

    order = [
        "descriptive_per_question",
        "predictive_per_option",
        "predictive_per_question",
        "explanatory_per_option",
        "explanatory_per_question",
        "counterfactual_per_option",
        "counterfactual_per_question",
        "overall_per_question",
    ]

    for key in order:
        value = metrics.get(key, float("nan"))
        print(f"{key:32}: {value * 100:7.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        choices=["raw", "world"],
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
    )
    args = parser.parse_args()

    seed_everything()

    train_data = load_json("train")
    vocab = build_vocab(train_data)
    answer_vocab = build_answer_vocab(train_data)

    rep_dir = CACHE_ROOT / args.representation
    train_meta_path = rep_dir / "train_meta.json"

    with train_meta_path.open("r") as f:
        train_meta = json.load(f)

    video_dim = train_meta["feature_dim"]

    train_dataset = CLEVRERQADataset(
        "train",
        args.representation,
        vocab,
        answer_vocab,
    )
    val_dataset = CLEVRERQADataset(
        "validation",
        args.representation,
        vocab,
        answer_vocab,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = QAProbe(
        video_dim=video_dim,
        vocab_size=len(vocab),
        num_answers=len(answer_vocab),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)

        progress = (
            step - warmup_steps
        ) / max(total_steps - warmup_steps, 1)

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    output_dir = OUTPUT_ROOT / args.representation
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "vocab.json").open("w") as f:
        json.dump(vocab, f, indent=2)

    with (output_dir / "answer_vocab.json").open("w") as f:
        json.dump(answer_vocab, f, indent=2)

    best_score = -1.0
    global_step = 0

    print("=" * 100)
    print(f"CLEVRER QA TRAINING: {args.representation.upper()}")
    print("=" * 100)
    print("Device:", DEVICE)
    print("Video dim:", video_dim)
    print("Vocabulary:", len(vocab))
    print("Descriptive answers:", len(answer_vocab))
    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Trainable parameters:", sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    ))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []

        for step, batch in enumerate(train_loader):
            video = batch["video"].to(
                DEVICE,
                non_blocking=True,
            )
            text_ids = batch["text_ids"].to(
                DEVICE,
                non_blocking=True,
            )
            text_mask = batch["text_mask"].to(
                DEVICE,
                non_blocking=True,
            )
            task = batch["task"].to(
                DEVICE,
                non_blocking=True,
            )
            target = batch["target"].to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            desc_logits, binary_logits = model(
                video,
                text_ids,
                text_mask,
            )

            loss = compute_loss(
                desc_logits,
                binary_logits,
                task,
                target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            running.append(float(loss))
            global_step += 1

            if step % 200 == 0:
                print(
                    f"epoch={epoch:02d} "
                    f"step={step:05d}/{len(train_loader):05d} "
                    f"loss={np.mean(running[-200:]):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        metrics = evaluate(
            model,
            val_loader,
        )

        print_metrics(
            metrics,
            f"VALIDATION EPOCH {epoch}",
        )

        score = metrics["overall_per_question"]

        ckpt = {
            "representation": args.representation,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": {
                "video_dim": video_dim,
                "vocab_size": len(vocab),
                "num_answers": len(answer_vocab),
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "ff_dim": FF_DIM,
                "dropout": DROPOUT,
                "max_text_len": MAX_TEXT_LEN,
                "max_video_steps": MAX_VIDEO_STEPS,
                "seed": SEED,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
        }

        torch.save(
            ckpt,
            output_dir / "last.pt",
        )

        if score > best_score:
            best_score = score
            torch.save(
                ckpt,
                output_dir / "best.pt",
            )
            print(
                f"Saved new best checkpoint: "
                f"{best_score * 100:.2f}% overall per-question"
            )

    print()
    print("Training complete.")
    print("Best overall per-question:", f"{best_score * 100:.2f}%")
    print("Checkpoint:", output_dir / "best.pt")


if __name__ == "__main__":
    main()
