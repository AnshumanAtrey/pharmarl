"""Generate publication-quality plots from eval JSON files.

Reads one or more eval JSON files (output of `eval_with_ci.py`) and produces:
  1. Per-target bar chart with error bars (mean ± std)
  2. Overall comparison bar chart (one bar per policy)
  3. Box plot showing distribution per target × policy

All plots are saved as PNG (300 DPI), with labeled axes, embedded in README.

Usage:
    python -m examples.plot_results \
        --inputs runs/random.json runs/scripted.json runs/trained_qwen.json \
        --output-dir docs/plots/

The README's Results section embeds these plots:
    docs/plots/per_target_comparison.png
    docs/plots/overall_comparison.png
    docs/plots/distribution_box.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Lazy import — matplotlib is the only heavy dep; keep the test-time import light
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    return plt


def load_eval(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def plot_per_target(evals: List[Dict[str, Any]], output: str) -> None:
    plt = _setup_matplotlib()
    targets = list(evals[0]["results"].keys())
    n_policies = len(evals)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / n_policies
    x_positions = list(range(len(targets)))

    colors = ["#888888", "#444444", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, ev in enumerate(evals):
        means = [ev["results"][t]["mean"] for t in targets]
        stds = [ev["results"][t]["std"] for t in targets]
        offset = (i - (n_policies - 1) / 2) * bar_width
        ax.bar(
            [x + offset for x in x_positions],
            means,
            bar_width,
            yerr=stds,
            label=ev["policy"],
            color=colors[i % len(colors)],
            capsize=4,
            alpha=0.85,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(targets)
    ax.set_xlabel("Target")
    ax.set_ylabel("Cumulative reward (mean ± std, N=10 episodes)")
    ax.set_title("PharmaRL — policy comparison per target")
    ax.legend(loc="best", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output}")


def plot_overall_comparison(evals: List[Dict[str, Any]], output: str) -> None:
    plt = _setup_matplotlib()

    policies = [ev["policy"] for ev in evals]
    means = [ev["overall"]["mean"] for ev in evals]
    stds = [ev["overall"]["std"] for ev in evals]

    sort_idx = sorted(range(len(means)), key=lambda i: means[i])
    policies = [policies[i] for i in sort_idx]
    means = [means[i] for i in sort_idx]
    stds = [stds[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4" if p == "trained_qwen" else "#888888" for p in policies]
    ax.barh(policies, means, xerr=stds, color=colors, alpha=0.85, capsize=5)
    ax.set_xlabel("Cumulative reward (mean ± std)")
    ax.set_title("PharmaRL — overall policy ranking")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output}")


def plot_distribution_box(evals: List[Dict[str, Any]], output: str) -> None:
    plt = _setup_matplotlib()
    targets = list(evals[0]["results"].keys())

    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4.5), sharey=True)
    if len(targets) == 1:
        axes = [axes]

    for j, target in enumerate(targets):
        data = [ev["results"][target]["episodes"] for ev in evals]
        labels = [ev["policy"] for ev in evals]
        axes[j].boxplot(data, labels=labels, showmeans=True, meanline=True)
        axes[j].set_title(f"{target}")
        axes[j].grid(True, axis="y", alpha=0.3)
        axes[j].tick_params(axis="x", labelrotation=30)
        if j == 0:
            axes[j].set_ylabel("Cumulative reward (per episode)")

    fig.suptitle("PharmaRL — reward distribution per target")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {output}")


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Eval JSON files (output of eval_with_ci.py)")
    p.add_argument("--output-dir", default="docs/plots/")
    args = p.parse_args(argv)

    evals = [load_eval(path) for path in args.inputs]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    plot_per_target(evals, f"{args.output_dir}/per_target_comparison.png")
    plot_overall_comparison(evals, f"{args.output_dir}/overall_comparison.png")
    plot_distribution_box(evals, f"{args.output_dir}/distribution_box.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
