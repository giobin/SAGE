"""Aggregate run metrics into the numbers reported in the paper.

Each run writes metrics.jsonl to its Hydra output directory. Runs are grouped by
experiment, environment and any non-seed overrides, then summarized over seeds:
peak and final evaluation return (Tables 1-8) and VLM calls (Table 9).

Usage: python scripts/summarize_results.py [multirun/ outputs/ ...]
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


def mean_std(values):
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.3f} ({std:.3f})"


def load_run(metrics_path: Path):
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    hydra_dir = metrics_path.parent / ".hydra"
    config = yaml.safe_load((hydra_dir / "config.yaml").read_text())
    overrides = yaml.safe_load((hydra_dir / "overrides.yaml").read_text()) or []
    returns = [r["eval_return"] for r in rows]
    return {
        "exp_name": config["exp_name"],
        "env_id": config["env"]["env_id"],
        "overrides": {o for o in overrides if not o.startswith(("seed=", "env="))},
        "peak": max(returns),
        "final": returns[-1],
        "steps": rows[-1]["global_step"],
        "vlm_calls": rows[-1]["vlm_calls"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="*", default=["multirun"], help="Directories to search for metrics.jsonl")
    args = parser.parse_args()

    runs = [run for root in args.dirs for path in sorted(Path(root).rglob("metrics.jsonl")) if (run := load_run(path))]

    # Overrides shared by all runs of an experiment are not shown; the rest identify the variant
    common = {}
    for run in runs:
        common[run["exp_name"]] = common.get(run["exp_name"], run["overrides"]) & run["overrides"]
    groups = defaultdict(list)
    for run in runs:
        variant = ",".join(sorted(run["overrides"] - common[run["exp_name"]]))
        groups[(run["exp_name"], run["env_id"], variant)].append(run)

    header = ["experiment", "env", "variant", "seeds", "peak return", "final return", "VLM calls", "query rate"]
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for (exp_name, env_id, variant), runs in sorted(groups.items()):
        calls = [r["vlm_calls"] for r in runs]
        rate = statistics.mean(r["vlm_calls"] / r["steps"] for r in runs if r["steps"] > 0) if any(r["steps"] for r in runs) else 0.0
        print(" | ".join([
            exp_name, env_id, variant or "-", str(len(runs)),
            mean_std([r["peak"] for r in runs]),
            mean_std([r["final"] for r in runs]),
            f"{statistics.mean(calls):,.0f} ± {statistics.stdev(calls) if len(calls) > 1 else 0.0:,.0f}",
            f"{100 * rate:.1f}%",
        ]))


if __name__ == "__main__":
    main()
