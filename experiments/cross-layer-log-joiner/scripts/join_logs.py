#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "test_runner_unavailable": {
        "pattern": r"No module named pytest|pytest is unavailable|command not found: pytest",
        "recommendation": "Mark fallback verification explicitly and capture the failed test runner command as structured evidence.",
    },
    "python_path_friction": {
        "pattern": r"`python` is not on PATH|command not found: python\b|python is not on PATH",
        "recommendation": "Normalize generated commands around detected interpreters, or record interpreter discovery in the run ledger.",
    },
    "codex_hook_config_drift": {
        "pattern": r"codex_hooks.*deprecated|\\[features\\]\\.codex_hooks|\\[features\\] hooks|--enable hooks",
        "recommendation": "Keep generated Codex hook config aligned with current Codex feature flags.",
    },
    "after_agent_hook_failure": {
        "pattern": r"after_agent hook failed|legacy_notify error=No such file or directory|legacy_notify",
        "recommendation": "Capture hook execution failures as first-class checkpoint/run health events.",
    },
    "automatic_capture_failure": {
        "pattern": r"Automatic capture did not create a checkpoint|Automatic checkpoint capture failed|automatic Codex capture failed|Automatic capture failing",
        "recommendation": "Expose automatic-capture success/failure in `entire status` and surface remediation commands.",
    },
    "hosted_auth_dependency": {
        "pattern": r"not authenticated|Run 'entire login'|hosted auth",
        "recommendation": "Distinguish local checkpoint inspection from hosted search/activity features in docs and CLI errors.",
    },
    "manual_attach_required": {
        "pattern": r"Manual session attachment worked|manual `entire session attach` worked|manual attach worked|session attach",
        "recommendation": "Treat manual attach as a fallback path but not the default happy path for autonomous systems.",
    },
}

SOURCE_LAYERS = {
    "entire_checkpoint_explain.txt": "entire_checkpoint",
    "entire_status.txt": "product_status",
    "entire_search_top_k_unique.txt": "cli_search",
    "entire_direct_test_output.txt": "test_output",
    "checkpoint_failure_report.json": "derived_failure_report",
}


def snippets(pattern, text, window=170):
    out = []
    for match in re.finditer(pattern, text, flags=re.I):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        out.append(" ".join(text[start:end].split()))
    return out


def read_inputs(inputs):
    records = {}
    for path in sorted(inputs.iterdir()):
        if path.is_file():
            records[path.name] = {
                "layer": SOURCE_LAYERS.get(path.name, "unknown"),
                "text": path.read_text(errors="replace"),
            }
    return records


def main():
    parser = argparse.ArgumentParser(description="Join Entire checkpoint evidence with external logs/finding files.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    sources = read_inputs(args.inputs)
    checkpoint_text = sources.get("entire_checkpoint_explain.txt", {}).get("text", "")
    joined = []
    checkpoint_visible = 0
    external_only = 0

    for category, spec in PATTERNS.items():
        hits = []
        for name, record in sources.items():
            found = snippets(spec["pattern"], record["text"])
            if found:
                hits.append(
                    {
                        "source": name,
                        "layer": record["layer"],
                        "examples": found[:3],
                    }
                )
        if not hits:
            continue
        in_checkpoint = bool(re.search(spec["pattern"], checkpoint_text, flags=re.I))
        checkpoint_visible += int(in_checkpoint)
        external_only += int(not in_checkpoint)
        joined.append(
            {
                "category": category,
                "checkpoint_visible": in_checkpoint,
                "layers": sorted({hit["layer"] for hit in hits}),
                "source_count": len(hits),
                "recommendation": spec["recommendation"],
                "hits": hits,
            }
        )

    report = {
        "summary": {
            "failure_categories": len(joined),
            "checkpoint_visible_categories": checkpoint_visible,
            "external_only_categories": external_only,
            "source_files": list(sources.keys()),
        },
        "categories": joined,
    }
    (args.results / "joined_failure_ledger.json").write_text(json.dumps(report, indent=2))

    print(f"Wrote cross-layer failure ledger with {len(joined)} categories")


if __name__ == "__main__":
    main()
