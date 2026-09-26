#!/usr/bin/env python3
"""Plot task-generation and reward success rates from an evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


TASK_COLOR = "#3B82C4"
REWARD_COLOR = "#E58B35"
TEXT_COLOR = "#263442"
GRID_COLOR = "#D9E0E6"
TITLE_SIZE = 28
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 22
MODEL_LABEL_SIZE = 24
LEGEND_SIZE = 22
VALUE_LABEL_SIZE = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped task-generation and reward success rates from "
            "evaluate_gpt_reward_models.py results.json."
        )
    )
    parser.add_argument("results", type=Path, help="Path to results.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <results-dir>/model_success_rates.png)",
    )
    parser.add_argument(
        "--title",
        default="GPT Model Evaluation Success Rates",
        help="Plot title",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output resolution")
    return parser.parse_args()


def load_summary(results_path: Path) -> list[dict[str, Any]]:
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload.get("summary")
    if not isinstance(summary, list) or not summary:
        raise ValueError(f"{results_path} has no non-empty 'summary' list")

    required = {
        "model",
        "task_generation_success_rate",
        "reward_success_rate",
    }
    for index, row in enumerate(summary):
        if not isinstance(row, dict):
            raise ValueError(f"summary[{index}] is not an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"summary[{index}] is missing: {', '.join(sorted(missing))}"
            )
        for key in ("task_generation_success_rate", "reward_success_rate"):
            rate = row[key]
            if not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
                raise ValueError(
                    f"summary[{index}].{key} must be between 0 and 1, got {rate!r}"
                )
    return summary


def add_value_labels(axis: plt.Axes, bars: Any) -> None:
    for bar in bars:
        rate = bar.get_height()
        axis.annotate(
            f"{rate:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, rate),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_SIZE,
            color=TEXT_COLOR,
        )


def plot_success_rates(
    summary: list[dict[str, Any]], output_path: Path, title: str, dpi: int
) -> None:
    models = [str(row["model"]) for row in summary]
    task_rates = [float(row["task_generation_success_rate"]) for row in summary]
    reward_rates = [float(row["reward_success_rate"]) for row in summary]

    positions = np.arange(len(models))
    bar_width = 0.36
    figure_width = max(13.0, 2.35 * len(models))
    figure, axis = plt.subplots(
        figsize=(figure_width, 7.4), constrained_layout=True, facecolor="white"
    )

    task_bars = axis.bar(
        positions - bar_width / 2,
        task_rates,
        bar_width,
        label="Feasible-task selection",
        color=TASK_COLOR,
    )
    reward_bars = axis.bar(
        positions + bar_width / 2,
        reward_rates,
        bar_width,
        label="Reward evaluation",
        color=REWARD_COLOR,
    )

    axis.set_title(title, fontsize=TITLE_SIZE, pad=58)
    axis.set_xlabel("Model", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    axis.set_ylabel("Success rate", fontsize=AXIS_LABEL_SIZE, labelpad=12)
    axis.set_xticks(
        positions,
        models,
        fontsize=MODEL_LABEL_SIZE,
    )
    axis.set_ylim(0, 1.12)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(
        axis="x", colors=TEXT_COLOR, labelsize=MODEL_LABEL_SIZE, width=1.2
    )
    axis.tick_params(
        axis="y", colors=TEXT_COLOR, labelsize=TICK_LABEL_SIZE, width=1.2
    )
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
    )

    add_value_labels(axis, task_bars)
    add_value_labels(axis, reward_bars)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_path = args.output or args.results.with_name("model_success_rates.png")
    summary = load_summary(args.results)
    plot_success_rates(summary, output_path, args.title, args.dpi)
    print(f"Saved success-rate plot to {output_path}")


if __name__ == "__main__":
    main()
