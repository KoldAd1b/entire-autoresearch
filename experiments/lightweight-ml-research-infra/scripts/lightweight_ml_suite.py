#!/usr/bin/env python3
import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetCard:
    name: str
    source: str
    task: str
    samples: int
    features: str
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
    epochs: int
    learning_rate: float


def one_hot(y, classes):
    out = np.zeros((len(y), classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def accuracy(logits, y):
    return float((logits.argmax(axis=1) == y).mean())


def cross_entropy(probs, y_onehot):
    return float(-(y_onehot * np.log(probs + 1e-8)).sum(axis=1).mean())


def make_batches(x, y, batch_size, rng):
    indices = rng.permutation(len(x))
    for start in range(0, len(x), batch_size):
        batch = indices[start : start + batch_size]
        yield x[batch], y[batch]


def train_mlp(name, x, y, classes, hidden, epochs, lr, batch_size, seed):
    rng = np.random.default_rng(seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    in_dim = x_train.shape[1]
    w1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2 / hidden), size=(hidden, classes)).astype(np.float32)
    b2 = np.zeros(classes, dtype=np.float32)
    y_train_oh = one_hot(y_train, classes)
    y_test_oh = one_hot(y_test, classes)

    t0 = time.time()
    history = []
    for epoch in range(epochs):
        for xb, yb in make_batches(x_train, y_train_oh, batch_size, rng):
            z1 = xb @ w1 + b1
            h = np.maximum(z1, 0)
            logits = h @ w2 + b2
            probs = softmax(logits)

            dlogits = (probs - yb) / len(xb)
            dw2 = h.T @ dlogits
            db2 = dlogits.sum(axis=0)
            dh = dlogits @ w2.T
            dz1 = dh * (z1 > 0)
            dw1 = xb.T @ dz1
            db1 = dz1.sum(axis=0)

            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2

        train_logits = np.maximum(x_train @ w1 + b1, 0) @ w2 + b2
        test_logits = np.maximum(x_test @ w1 + b1, 0) @ w2 + b2
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": cross_entropy(softmax(train_logits), y_train_oh),
                "test_loss": cross_entropy(softmax(test_logits), y_test_oh),
                "train_accuracy": accuracy(train_logits, y_train),
                "test_accuracy": accuracy(test_logits, y_test),
            }
        )

    params = w1.size + b1.size + w2.size + b2.size
    return {
        "name": name,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "params": int(params),
        "seconds": round(time.time() - t0, 4),
        "history": history,
        "final": history[-1],
    }


def conv_forward(x, filters, bias):
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))
    patches = windows[:, 0]
    return np.einsum("nijab,kab->nkij", patches, filters) + bias[None, :, None, None], patches


def train_tiny_cnn(x_images, y, epochs, lr, batch_size, seed):
    rng = np.random.default_rng(seed)
    x = (x_images.astype(np.float32) / 16.0)[:, None, :, :]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    classes = 10
    filters = rng.normal(0, 0.15, size=(8, 3, 3)).astype(np.float32)
    conv_bias = np.zeros(8, dtype=np.float32)
    w = rng.normal(0, 0.15, size=(8 * 3 * 3, classes)).astype(np.float32)
    b = np.zeros(classes, dtype=np.float32)
    y_train_oh = one_hot(y_train, classes)
    y_test_oh = one_hot(y_test, classes)

    t0 = time.time()
    history = []
    for epoch in range(epochs):
        for xb, yb in make_batches(x_train, y_train_oh, batch_size, rng):
            conv, patches = conv_forward(xb, filters, conv_bias)
            relu = np.maximum(conv, 0)
            pooled = relu.reshape(len(xb), 8, 3, 2, 3, 2).mean(axis=(3, 5))
            flat = pooled.reshape(len(xb), -1)
            logits = flat @ w + b
            probs = softmax(logits)

            dlogits = (probs - yb) / len(xb)
            dw = flat.T @ dlogits
            db = dlogits.sum(axis=0)
            dflat = dlogits @ w.T
            dpooled = dflat.reshape(len(xb), 8, 3, 3)
            drelu = np.repeat(np.repeat(dpooled[:, :, :, :, None], 2, axis=4), 2, axis=3) / 4.0
            drelu = drelu.reshape(len(xb), 8, 6, 6)
            dconv = drelu * (conv > 0)
            dfilters = np.einsum("nkij,nijab->kab", dconv, patches)
            dconv_bias = dconv.sum(axis=(0, 2, 3))

            filters -= lr * dfilters
            conv_bias -= lr * dconv_bias
            w -= lr * dw
            b -= lr * db

        train_logits = cnn_logits(x_train, filters, conv_bias, w, b)
        test_logits = cnn_logits(x_test, filters, conv_bias, w, b)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": cross_entropy(softmax(train_logits), y_train_oh),
                "test_loss": cross_entropy(softmax(test_logits), y_test_oh),
                "train_accuracy": accuracy(train_logits, y_train),
                "test_accuracy": accuracy(test_logits, y_test),
            }
        )

    params = filters.size + conv_bias.size + w.size + b.size
    return {
        "name": "numpy_cnn_digits",
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "params": int(params),
        "seconds": round(time.time() - t0, 4),
        "history": history,
        "final": history[-1],
    }


def cnn_logits(x, filters, conv_bias, w, b):
    conv, _ = conv_forward(x, filters, conv_bias)
    relu = np.maximum(conv, 0)
    pooled = relu.reshape(len(x), 8, 3, 2, 3, 2).mean(axis=(3, 5))
    flat = pooled.reshape(len(x), -1)
    return flat @ w + b


def write_outputs(out_dir, dataset_cards, model_cards, runs):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_cards.json").write_text(
        json.dumps([asdict(card) for card in dataset_cards], indent=2)
    )
    (out_dir / "model_cards.json").write_text(
        json.dumps([asdict(card) for card in model_cards], indent=2)
    )
    (out_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_by": "lightweight_ml_suite.py",
                "purpose": "Research-infrastructure artifact for Entire/Tigris Track 1 diligence",
                "runs": runs,
            },
            indent=2,
        )
    )

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "epoch",
                "train_loss",
                "test_loss",
                "train_accuracy",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        for run in runs:
            for row in run["history"]:
                writer.writerow({"run": run["name"], **row})

    lines = [
        "# Lightweight ML Experiment Summary",
        "",
        "## Why This Matters",
        "",
        "This turns the Entire evaluation into a research-infrastructure test: real datasets, real training runs, metrics, dataset cards, model cards, and reproducibility metadata.",
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
                f"- Final train accuracy: {final['train_accuracy']:.4f}",
                f"- Final test accuracy: {final['test_accuracy']:.4f}",
                f"- Final test loss: {final['test_loss']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Infrastructure Takeaway",
            "",
            "For a BSV-style product diligence writeup, these artifacts show the next product layer Entire could support: checkpoint-linked ML experiment provenance. The missing piece is a joined ledger that binds the agent transcript, code commit, dataset card, model card, metrics, and artifact storage pointer into one reproducible record.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Train lightweight NumPy MLP/CNN baselines and write research artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    digits = load_digits()
    cancer = load_breast_cancer()

    dataset_cards = [
        DatasetCard(
            name="sklearn_digits",
            source="scikit-learn built-in load_digits",
            task="10-class image classification",
            samples=len(digits.data),
            features="8x8 grayscale images",
            classes=10,
            train_samples=int(len(digits.data) * 0.75),
            test_samples=len(digits.data) - int(len(digits.data) * 0.75),
        ),
        DatasetCard(
            name="sklearn_breast_cancer",
            source="scikit-learn built-in load_breast_cancer",
            task="binary tabular classification",
            samples=len(cancer.data),
            features=f"{cancer.data.shape[1]} numeric features",
            classes=2,
            train_samples=int(len(cancer.data) * 0.75),
            test_samples=len(cancer.data) - int(len(cancer.data) * 0.75),
        ),
    ]

    runs = [
        train_mlp(
            "numpy_mlp_breast_cancer",
            cancer.data.astype(np.float32),
            cancer.target.astype(int),
            classes=2,
            hidden=32,
            epochs=args.epochs,
            lr=0.025,
            batch_size=64,
            seed=args.seed,
        ),
        train_mlp(
            "numpy_mlp_digits",
            digits.data.astype(np.float32) / 16.0,
            digits.target.astype(int),
            classes=10,
            hidden=64,
            epochs=args.epochs,
            lr=0.08,
            batch_size=128,
            seed=args.seed,
        ),
        train_tiny_cnn(
            digits.images,
            digits.target.astype(int),
            epochs=args.epochs,
            lr=0.04,
            batch_size=128,
            seed=args.seed,
        ),
    ]

    model_cards = [
        ModelCard("numpy_mlp_breast_cancer", "sklearn_breast_cancer", "30 -> 32 ReLU -> 2 softmax", runs[0]["params"], "mini-batch SGD", args.epochs, 0.025),
        ModelCard("numpy_mlp_digits", "sklearn_digits", "64 -> 64 ReLU -> 10 softmax", runs[1]["params"], "mini-batch SGD", args.epochs, 0.08),
        ModelCard("numpy_cnn_digits", "sklearn_digits", "8 trainable 3x3 filters -> ReLU -> 2x2 avg pool -> 10 softmax", runs[2]["params"], "mini-batch SGD", args.epochs, 0.04),
    ]
    write_outputs(args.out_dir, dataset_cards, model_cards, runs)
    print(f"Wrote ML research artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
