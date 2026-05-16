#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def extract_entire_signals(text):
    return {
        "checkpoint_id": re.search(r"Checkpoint\s+([a-f0-9]{12,})", text, flags=re.I).group(1)
        if re.search(r"Checkpoint\s+([a-f0-9]{12,})", text, flags=re.I)
        else "unknown",
        "tool_calls": len(re.findall(r"\[Tool\]", text)),
        "assistant_turns": len(re.findall(r"\[Assistant\]", text)),
        "user_turns": len(re.findall(r"\[User\]", text)),
        "test_mentions": len(re.findall(r"pytest|test_|assert|passed|failed", text, flags=re.I)),
        "failure_mentions": len(re.findall(r"failed|error|unavailable|fallback|not authenticated|No module", text, flags=re.I)),
        "has_commit_linkage": bool(re.search(r"Commit|Entire-Checkpoint", text, flags=re.I)),
    }


def artifact_inventory(inputs_dir):
    return [
        {
            "path": str(path.relative_to(inputs_dir.parent)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(inputs_dir.rglob("*"))
        if path.is_file()
    ]


def main():
    parser = argparse.ArgumentParser(description="Build a joined autonomous-research provenance ledger.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    entire_text = (args.inputs / "entire_checkpoint_explain.txt").read_text()
    manifest = read_json(args.inputs / "autonomous_run_manifest.json")
    decisions = read_jsonl(args.inputs / "decisions.jsonl")
    trajectory = read_csv(args.inputs / "trajectory.csv")
    scorecard = read_json(args.inputs / "entire_fit_scorecard.json")

    baseline_acc = float(manifest["baseline_accuracy"])
    best_acc = float(manifest["best_accuracy"])
    best_trial = manifest["best_trial"]
    best_row = next(row for row in trajectory if row["trial_id"] == best_trial)
    entire_signals = extract_entire_signals(entire_text)

    ledger = {
        "ledger_type": "autonomous_research_provenance",
        "thesis": "Entire is strong as an agent-trace spine, but autonomous research requires joined metrics, decisions, environment metadata, datasets, and artifact pointers.",
        "entire_checkpoint": entire_signals,
        "autonomous_run": {
            "system": manifest["system"],
            "objective": manifest["objective"],
            "device": manifest["device"],
            "environment": manifest["environment"],
            "baseline_trial": manifest["baseline_trial"],
            "best_trial": best_trial,
            "baseline_accuracy": baseline_acc,
            "best_accuracy": best_acc,
            "absolute_improvement": best_acc - baseline_acc,
            "trial_count": len(trajectory),
            "decision_count": len(decisions),
        },
        "best_trial": {
            "trial_id": best_trial,
            "model_type": best_row["model_type"],
            "hidden": int(best_row["hidden"]),
            "dropout": float(best_row["dropout"]),
            "learning_rate": float(best_row["learning_rate"]),
            "params": int(best_row["params"]),
            "test_accuracy": float(best_row["test_accuracy"]),
            "checkpoint": best_row["checkpoint"],
        },
        "entire_fit": {
            "entire_checkpoint_only_score": scorecard["entire_checkpoint_only_score"],
            "joined_research_system_score": scorecard["joined_research_system_score"],
            "max_score": scorecard["max_score"],
            "missing_from_entire_only": [
                row["requirement"]
                for row in scorecard["requirements"]
                if not row["entire_checkpoint_only"] and row["joined_research_system"]
            ],
        },
        "artifacts": artifact_inventory(args.inputs),
        "production_recommendations": [
            "Expose first-class artifact links from checkpoints to metrics, manifests, model checkpoints, and object-store URIs.",
            "Capture scheduler decisions as structured events, not only transcript prose.",
            "Separate agent trace, shell logs, and experiment logs while linking them under one research-run id.",
            "Add local/offline search over checkpoint and artifact metadata for private research workflows.",
            "Make hook integration health part of the run ledger because automatic capture reliability is critical.",
        ],
    }

    (args.results / "provenance_ledger.json").write_text(json.dumps(ledger, indent=2))

    missing = ", ".join(f"`{item}`" for item in ledger["entire_fit"]["missing_from_entire_only"])
    md = f"""# Autonomous Research Provenance Ledger

## Thesis

Entire works best as the agent-trace spine for autonomous research. It captures intent, transcript, tool calls, and commit linkage, but the full research record needs joined experiment artifacts.

## Run Summary

- System: `{ledger["autonomous_run"]["system"]}`
- Objective: {ledger["autonomous_run"]["objective"]}
- Trials: {ledger["autonomous_run"]["trial_count"]}
- Decisions: {ledger["autonomous_run"]["decision_count"]}
- Device: `{ledger["autonomous_run"]["device"]}`
- Baseline: `{ledger["autonomous_run"]["baseline_trial"]}` at {baseline_acc:.4f}
- Best: `{best_trial}` at {best_acc:.4f}
- Absolute improvement: {best_acc - baseline_acc:.4f}

## Best Trial

- Model: `{ledger["best_trial"]["model_type"]}`
- Hidden units: {ledger["best_trial"]["hidden"]}
- Dropout: {ledger["best_trial"]["dropout"]}
- Learning rate: {ledger["best_trial"]["learning_rate"]}
- Parameters: {ledger["best_trial"]["params"]}
- Checkpoint pointer: `{ledger["best_trial"]["checkpoint"]}`

## Entire Evidence

- Checkpoint id: `{entire_signals["checkpoint_id"]}`
- Tool calls: {entire_signals["tool_calls"]}
- Assistant turns: {entire_signals["assistant_turns"]}
- User turns: {entire_signals["user_turns"]}
- Test mentions: {entire_signals["test_mentions"]}
- Failure mentions: {entire_signals["failure_mentions"]}
- Commit linkage detected: {entire_signals["has_commit_linkage"]}

## Entire Fit

- Entire checkpoint only: {scorecard["entire_checkpoint_only_score"]}/{scorecard["max_score"]}
- Joined research record: {scorecard["joined_research_system_score"]}/{scorecard["max_score"]}
- Missing from Entire-only record: {missing}

## Recommendation

Entire should be evaluated as the agent-process provenance layer, not as a full ML experiment tracker. The product opportunity is to connect checkpoints to structured experiment artifacts: scheduler decisions, metrics, model checkpoints, dataset snapshots, environment metadata, and storage URIs.
"""
    (args.results / "provenance_ledger.md").write_text(md)

    claims = f"""# Submission-Ready Claims

1. I built a tiny autonomous research experimentation system, not just a static product demo.
2. The system ran {len(trajectory)} trials and improved validation accuracy from {baseline_acc:.2%} to {best_acc:.2%}.
3. Entire checkpoint-only evidence covered {scorecard["entire_checkpoint_only_score"]}/{scorecard["max_score"]} provenance needs.
4. A joined research ledger covered {scorecard["joined_research_system_score"]}/{scorecard["max_score"]}, showing the exact product gap.
5. My conclusion: Entire is compelling as the agent-trace layer, but it needs artifact linkage to become full autonomous research infrastructure.
"""
    (args.results / "submission_claims.md").write_text(claims)
    print(f"Wrote joined provenance ledger to {args.results}")


if __name__ == "__main__":
    main()
