#!/usr/bin/env python3
import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetCard:
    name: str
    source: str
    task: str
    samples: int
    shape: str
    classes: int
    train_samples: int
    test_samples: int


@dataclass
class ModelCard:
    name: str
    dataset: str
    architecture: str
    trainable_parameters: int
    optimizer: str
    loss: str
    epochs: int
    learning_rate: float
    checkpoint: str


class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_breast_cancer(seed, batch_size):
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return data, make_loader(x_train, y_train, batch_size, True), make_loader(x_test, y_test, batch_size, False)


def prepare_digits(seed, batch_size, image):
    data = load_digits()
    x = data.images[:, None, :, :] / 16.0 if image else data.data / 16.0
    x_train, x_test, y_train, y_test = train_test_split(
        x, data.target, test_size=0.25, random_state=seed, stratify=data.target
    )
    return data, make_loader(x_train, y_train, batch_size, True), make_loader(x_test, y_test, batch_size, False)


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(x)
    return total_loss / total, correct / total


def train_run(name, model, train_loader, test_loader, device, epochs, lr, checkpoint_path):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
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
                "run": name,
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "state_dict": model.cpu().state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    return {
        "name": name,
        "params": count_params(model),
        "seconds": round(time.time() - t0, 4),
        "checkpoint": str(checkpoint_path),
        "history": history,
        "final": history[-1],
    }


def write_artifacts(out_dir, dataset_cards, model_cards, runs, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "torch_dataset_cards.json").write_text(json.dumps([asdict(card) for card in dataset_cards], indent=2))
    (out_dir / "torch_model_cards.json").write_text(json.dumps([asdict(card) for card in model_cards], indent=2))
    (out_dir / "torch_run_manifest.json").write_text(
        json.dumps(
            {
                "created_by": "torch_ml_suite.py",
                "device": str(device),
                "purpose": "PyTorch MLP/CNN artifacts for Entire-as-research-infrastructure diligence",
                "runs": runs,
            },
            indent=2,
        )
    )
    with (out_dir / "torch_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy"],
        )
        writer.writeheader()
        for run in runs:
            writer.writerows(run["history"])

    lines = [
        "# PyTorch Lightweight ML Summary",
        "",
        f"- Device: `{device}`",
        f"- Runs: {len(runs)}",
        "",
        "## Results",
        "",
    ]
    for run in runs:
        final = run["final"]
        lines.extend(
            [
                f"### {run['name']}",
                "",
                f"- Parameters: {run['params']}",
                f"- Runtime seconds: {run['seconds']}",
                f"- Checkpoint: `{run['checkpoint']}`",
                f"- Final train accuracy: {final['train_accuracy']:.4f}",
                f"- Final test accuracy: {final['test_accuracy']:.4f}",
                f"- Final test loss: {final['test_loss']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Submission Angle",
            "",
            "These runs create the kind of concrete artifact trail that a research-infrastructure version of Entire should preserve: dataset choice, architecture, hyperparameters, metrics, checkpoints, and the agent transcript that produced them.",
        ]
    )
    (out_dir / "torch_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Train lightweight PyTorch MLP/CNN baselines for research-infrastructure diligence.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device()
    checkpoints = args.out_dir / "checkpoints"

    cancer_data, cancer_train, cancer_test = prepare_breast_cancer(args.seed, args.batch_size)
    digits_data, digits_train_flat, digits_test_flat = prepare_digits(args.seed, args.batch_size, image=False)
    _, digits_train_img, digits_test_img = prepare_digits(args.seed, args.batch_size, image=True)

    run_specs = [
        ("torch_mlp_breast_cancer", TabularMLP(cancer_data.data.shape[1], 64, 2), cancer_train, cancer_test, 0.0015),
        ("torch_mlp_digits", DigitMLP(), digits_train_flat, digits_test_flat, 0.001),
        ("torch_cnn_digits", DigitCNN(), digits_train_img, digits_test_img, 0.001),
    ]

    runs = []
    for name, model, train_loader, test_loader, lr in run_specs:
        runs.append(
            train_run(
                name,
                model,
                train_loader,
                test_loader,
                device,
                args.epochs,
                lr,
                checkpoints / f"{name}.pt",
            )
        )

    dataset_cards = [
        DatasetCard("sklearn_breast_cancer", "scikit-learn built-in load_breast_cancer", "binary tabular classification", len(cancer_data.data), str(cancer_data.data.shape), 2, len(cancer_train.dataset), len(cancer_test.dataset)),
        DatasetCard("sklearn_digits", "scikit-learn built-in load_digits", "10-class image classification", len(digits_data.data), "1797 x 8 x 8", 10, len(digits_train_img.dataset), len(digits_test_img.dataset)),
    ]
    model_cards = [
        ModelCard("torch_mlp_breast_cancer", "sklearn_breast_cancer", "30 -> 64 ReLU -> 64 ReLU -> 2", runs[0]["params"], "AdamW", "CrossEntropyLoss", args.epochs, 0.0015, runs[0]["checkpoint"]),
        ModelCard("torch_mlp_digits", "sklearn_digits", "64 -> 128 ReLU dropout -> 10", runs[1]["params"], "AdamW", "CrossEntropyLoss", args.epochs, 0.001, runs[1]["checkpoint"]),
        ModelCard("torch_cnn_digits", "sklearn_digits", "Conv16 -> pool -> Conv32 -> MLP", runs[2]["params"], "AdamW", "CrossEntropyLoss", args.epochs, 0.001, runs[2]["checkpoint"]),
    ]
    write_artifacts(args.out_dir, dataset_cards, model_cards, runs, device)
    print(f"Wrote PyTorch ML artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
