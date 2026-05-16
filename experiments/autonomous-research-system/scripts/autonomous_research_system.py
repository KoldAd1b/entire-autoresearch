#!/usr/bin/env python3
import argparse
import csv
import json
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrialConfig:
    trial_id: str
    model_type: str
    hidden: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    rationale: str


class DigitMLP(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


class DigitCNN(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.head(self.features(x))


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-repo"


def package_versions():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
    }


def load_data(batch_size, model_type, seed):
    digits = load_digits()
    x = digits.images[:, None, :, :] / 16.0 if model_type == "cnn" else digits.data / 16.0
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    train = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
        {"name": "sklearn_digits", "samples": len(digits.data), "shape": "1797 x 8 x 8", "classes": 10},
    )


def make_model(config):
    if config.model_type == "cnn":
        return DigitCNN(config.hidden, config.dropout)
    return DigitMLP(config.hidden, config.dropout)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(x)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(x)
    return total_loss / total, total_correct / total


def train_trial(config, out_dir, device, seed):
    torch.manual_seed(seed)
    train_loader, test_loader, dataset = load_data(config.batch_size, config.model_type, seed)
    model = make_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    history = []
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_loss, train_acc = evaluate(model, train_loader, device, loss_fn)
        test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
        history.append(
            {
                "trial_id": config.trial_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }
        )
    checkpoint = out_dir / "checkpoints" / f"{config.trial_id}_{config.model_type}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trial_config": asdict(config),
            "state_dict": model.cpu().state_dict(),
            "history": history,
        },
        checkpoint,
    )
    final = history[-1]
    return {
        "trial_id": config.trial_id,
        "config": asdict(config),
        "dataset": dataset,
        "params": count_params(model),
        "runtime_seconds": round(time.time() - start, 4),
        "checkpoint": str(checkpoint),
        "final": final,
        "history": history,
    }


def next_config(trial_index, prior_results, epochs):
    seeds = [
        TrialConfig("trial_001", "mlp", 64, 0.00, 0.0010, 64, epochs, "Bootstrap low-capacity MLP baseline."),
        TrialConfig("trial_002", "mlp", 128, 0.10, 0.0010, 64, epochs, "Increase MLP capacity and add mild dropout."),
        TrialConfig("trial_003", "cnn", 64, 0.10, 0.0010, 64, epochs, "Switch to image-biased CNN architecture."),
    ]
    if trial_index <= len(seeds):
        return seeds[trial_index - 1]

    best = max(prior_results, key=lambda r: r["final"]["test_accuracy"])
    best_cfg = best["config"]
    if best_cfg["model_type"] == "cnn":
        hidden = min(192, int(best_cfg["hidden"] * 1.5))
        lr = best_cfg["learning_rate"] * (0.8 if trial_index % 2 == 0 else 1.2)
        dropout = 0.10 if best["final"]["train_accuracy"] - best["final"]["test_accuracy"] < 0.04 else 0.20
        rationale = "Exploit the current CNN winner, adjust capacity/lr based on generalization gap."
        return TrialConfig(f"trial_{trial_index:03d}", "cnn", hidden, dropout, lr, 64, epochs, rationale)

    rationale = "MLP is currently winning; test whether inductive bias from CNN improves validation accuracy."
    return TrialConfig(f"trial_{trial_index:03d}", "cnn", 96, 0.10, 0.0010, 64, epochs, rationale)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_entire_fit(entire_text, manifest):
    requirements = [
        ("intent", r"\[User\]|Intent", "Captures original research/developer intent."),
        ("transcript", r"Transcript|\[Assistant\]", "Captures agent reasoning/process transcript."),
        ("tool_calls", r"\[Tool\]|exec_command", "Captures commands/tool calls."),
        ("commit_linkage", r"Entire-Checkpoint|Commit", "Links checkpoint to commit context."),
        ("metrics", r"accuracy|loss|metrics|epoch", "Captures experiment metrics."),
        ("dataset_identity", r"dataset|sklearn_digits|load_digits", "Captures dataset identity and version."),
        ("artifact_pointers", r"\.pt|checkpoint|artifact|results/", "Captures model/artifact pointers."),
        ("environment", r"torch|python|platform|requirements|venv", "Captures environment/package metadata."),
        ("scheduler_decisions", r"trial_|rationale|policy|next experiment", "Captures autonomous scheduler decisions."),
        ("cross_layer_failures", r"failed|error|hook failed|unavailable|fallback", "Captures agent, shell, and experiment failures."),
    ]
    combined = entire_text + "\n" + json.dumps(manifest)
    rows = []
    entire_only_score = 0
    joined_score = 0
    for key, pattern, description in requirements:
        in_entire = bool(re.search(pattern, entire_text, flags=re.I))
        in_joined = bool(re.search(pattern, combined, flags=re.I))
        entire_only_score += int(in_entire)
        joined_score += int(in_joined)
        rows.append(
            {
                "requirement": key,
                "description": description,
                "entire_checkpoint_only": in_entire,
                "joined_research_system": in_joined,
            }
        )
    return rows, entire_only_score, joined_score


def main():
    parser = argparse.ArgumentParser(description="Run a tiny autonomous ML research loop and score Entire as provenance infrastructure.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--entire-explain", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    results = []
    decisions = []
    all_history = []

    for trial_idx in range(1, args.trials + 1):
        config = next_config(trial_idx, results, args.epochs)
        decisions.append(
            {
                "trial_id": config.trial_id,
                "decision_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "policy": "bootstrap_then_exploit_best",
                "rationale": config.rationale,
                "prior_best": max((r["final"]["test_accuracy"] for r in results), default=None),
                "config": asdict(config),
            }
        )
        result = train_trial(config, args.out_dir, device, args.seed + trial_idx)
        results.append(result)
        all_history.extend(result["history"])

    best = max(results, key=lambda r: r["final"]["test_accuracy"])
    baseline = results[0]
    manifest = {
        "system": "tiny_autonomous_research_system",
        "git_commit": git_commit(),
        "device": str(device),
        "environment": package_versions(),
        "objective": "maximize sklearn_digits validation accuracy under a short local compute budget",
        "baseline_trial": baseline["trial_id"],
        "best_trial": best["trial_id"],
        "baseline_accuracy": baseline["final"]["test_accuracy"],
        "best_accuracy": best["final"]["test_accuracy"],
        "absolute_improvement": best["final"]["test_accuracy"] - baseline["final"]["test_accuracy"],
        "trials": results,
    }

    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out_dir / "decisions.jsonl").write_text("\n".join(json.dumps(row) for row in decisions) + "\n")
    write_csv(
        args.out_dir / "trajectory.csv",
        [
            {
                "trial_id": r["trial_id"],
                "model_type": r["config"]["model_type"],
                "hidden": r["config"]["hidden"],
                "dropout": r["config"]["dropout"],
                "learning_rate": r["config"]["learning_rate"],
                "params": r["params"],
                "runtime_seconds": r["runtime_seconds"],
                "test_accuracy": r["final"]["test_accuracy"],
                "test_loss": r["final"]["test_loss"],
                "checkpoint": r["checkpoint"],
            }
            for r in results
        ],
        ["trial_id", "model_type", "hidden", "dropout", "learning_rate", "params", "runtime_seconds", "test_accuracy", "test_loss", "checkpoint"],
    )
    write_csv(
        args.out_dir / "epoch_metrics.csv",
        all_history,
        ["trial_id", "epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy"],
    )

    entire_text = args.entire_explain.read_text() if args.entire_explain.exists() else ""
    score_rows, entire_score, joined_score = score_entire_fit(entire_text, manifest)
    (args.out_dir / "entire_fit_scorecard.json").write_text(
        json.dumps(
            {
                "entire_checkpoint_only_score": entire_score,
                "joined_research_system_score": joined_score,
                "max_score": len(score_rows),
                "requirements": score_rows,
            },
            indent=2,
        )
    )

    print(f"Wrote autonomous research artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
