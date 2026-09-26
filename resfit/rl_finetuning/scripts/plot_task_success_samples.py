#!/usr/bin/env python3
"""Plot one success-rate and cumulative-sample figure per task."""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


SUCCESS_KEY = re.compile(r"^tasks/task_(\d+)_success_rate$")
PROBABILITY_KEY = re.compile(r"^tasks/task_(\d+)_probability$")


def _history_row(record: wandb_internal_pb2.Record) -> dict[str, object]:
    row: dict[str, object] = {}
    for item in record.history.item:
        key = "/".join(item.nested_key) if item.nested_key else item.key
        row[key] = json.loads(item.value_json)
    return row


def _read_task_points(
    wandb_files: list[Path],
) -> tuple[
    dict[int, list[tuple[int, float, int]]],
    dict[int, list[tuple[int, float]]],
    int,
]:
    # Store timestamp as part of the intermediate value so a later resumed
    # session wins if W&B contains two records for the same global step.
    by_task_and_step: dict[int, dict[int, tuple[float, float, int]]] = {}
    probability_by_task_and_step: dict[
        int, dict[int, tuple[float, float]]
    ] = {}
    run_last_step = 0

    for wandb_file in wandb_files:
        store = DataStore()
        store.open_for_scan(str(wandb_file))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue

            row = _history_row(record)
            if "_step" not in row:
                continue
            step = int(row["_step"])
            run_last_step = max(run_last_step, step)
            timestamp = float(row.get("_timestamp", 0.0))

            for key, value in row.items():
                match = SUCCESS_KEY.fullmatch(key)
                if match is None:
                    continue
                task_index = int(match.group(1))
                count_key = f"tasks/task_{task_index}_episode_count"
                attempts_key = f"tasks/task_{task_index}_attempts"
                if count_key in row:
                    sample_count = int(row[count_key])
                elif attempts_key in row:
                    sample_count = int(row[attempts_key])
                else:
                    raise RuntimeError(
                        f"Success-rate record for task {task_index} at step "
                        f"{step} has no episode/sample count"
                    )
                task_steps = by_task_and_step.setdefault(task_index, {})
                previous = task_steps.get(step)
                if previous is None or timestamp >= previous[0]:
                    task_steps[step] = (
                        timestamp,
                        float(value),
                        sample_count,
                    )

            for key, value in row.items():
                match = PROBABILITY_KEY.fullmatch(key)
                if match is None:
                    continue
                task_index = int(match.group(1))
                task_steps = probability_by_task_and_step.setdefault(
                    task_index, {}
                )
                previous = task_steps.get(step)
                if previous is None or timestamp >= previous[0]:
                    task_steps[step] = (timestamp, float(value))

    points = {
        task_index: [
            (step, values[1], values[2])
            for step, values in sorted(task_steps.items())
        ]
        for task_index, task_steps in sorted(by_task_and_step.items())
    }
    probability_points = {
        task_index: [
            (step, values[1])
            for step, values in sorted(task_steps.items())
        ]
        for task_index, task_steps in sorted(
            probability_by_task_and_step.items()
        )
    }
    return points, probability_points, run_last_step


def _infer_task_labels(session_dirs: list[Path], task_count: int) -> list[str]:
    marker = "[collector] task success stats: "
    for session_dir in session_dirs:
        output_log = session_dir / "files" / "output.log"
        if not output_log.is_file():
            continue
        with output_log.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if marker not in line:
                    continue
                try:
                    stats = ast.literal_eval(line.split(marker, 1)[1].strip())
                except (SyntaxError, ValueError):
                    continue
                if isinstance(stats, dict) and len(stats) == task_count:
                    return [str(task) for task in stats]
    return [f"task_{index}" for index in range(task_count)]


def _format_step(value: float, _position: int) -> str:
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def _plot_task(
    *,
    task_index: int,
    task_label: str,
    points: list[tuple[int, float, int]],
    probability_points: list[tuple[int, float]],
    run_last_step: int,
    sample_axis_max: int,
    output_path: Path,
) -> None:
    steps = [point[0] for point in points]
    success_rates = [point[1] for point in points]
    sample_counts = [point[2] for point in points]
    probability_steps = [point[0] for point in probability_points]
    probabilities = [point[1] for point in probability_points]

    # Extend the final state to the end of the run so all eight figures have
    # the same x-range and remain directly comparable.
    plot_steps = list(steps)
    plot_rates = list(success_rates)
    plot_counts = list(sample_counts)
    if plot_steps[-1] < run_last_step:
        plot_steps.append(run_last_step)
        plot_rates.append(plot_rates[-1])
        plot_counts.append(plot_counts[-1])
    plot_probability_steps = list(probability_steps)
    plot_probabilities = list(probabilities)
    if plot_probability_steps[-1] < run_last_step:
        plot_probability_steps.append(run_last_step)
        plot_probabilities.append(plot_probabilities[-1])

    figure, rate_axis = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
    count_axis = rate_axis.twinx()

    rate_color = "#62add8"
    probability_color = "#2a9d55"
    count_color = "#e07a1f"
    rate_axis.fill_between(
        plot_steps,
        plot_rates,
        step="post",
        color=rate_color,
        alpha=0.16,
        linewidth=0,
        label="Rolling success rate",
    )
    count_line = count_axis.step(
        plot_steps,
        plot_counts,
        where="post",
        color=count_color,
        linewidth=1.8,
        alpha=0.9,
        label="Cumulative samples",
    )[0]
    rate_line = rate_axis.step(
        plot_steps,
        plot_rates,
        where="post",
        color=rate_color,
        linewidth=2.2,
        label="Rolling success rate",
    )[0]
    probability_line = rate_axis.step(
        plot_probability_steps,
        plot_probabilities,
        where="post",
        color=probability_color,
        linewidth=1.8,
        linestyle="-",
        alpha=0.95,
        label="Logged task probability (all-task normalized)",
    )[0]
    rate_axis.scatter(
        steps,
        success_rates,
        color=rate_color,
        edgecolors="white",
        linewidths=0.45,
        s=18,
        zorder=3,
    )

    rate_axis.set_xlim(0, run_last_step)
    rate_axis.set_ylim(0, 1)
    count_axis.set_ylim(0, sample_axis_max)
    rate_axis.set_xlabel("Global training step")
    rate_axis.set_ylabel("Success rate / logged probability")
    count_axis.set_ylabel("Cumulative task samples", color=count_color)
    count_axis.tick_params(axis="y", colors=count_color)
    rate_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    rate_axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
    sample_locator = MaxNLocator(nbins=6, integer=True)
    sample_ticks = [
        tick
        for tick in sample_locator.tick_values(0, sample_axis_max)
        if 0 <= tick < sample_axis_max
    ]
    count_axis.set_yticks([*sample_ticks, sample_axis_max])
    rate_axis.grid(axis="both", color="#b0b0b0", alpha=0.28, linewidth=0.8)
    rate_axis.set_axisbelow(True)
    rate_axis.set_title(f"Task {task_index}: {task_label}", pad=12)

    rate_axis.legend(
        [
            Patch(facecolor=rate_color, edgecolor=rate_color, alpha=0.16),
            probability_line,
            count_line,
        ],
        [
            "Rolling success rate",
            "Logged task probability (all-task normalized)",
            "Cumulative samples",
        ],
        loc="upper left",
        frameon=False,
    )
    rate_axis.text(
        0.985,
        0.96,
        f"Final success rate: {success_rates[-1]:.1%}\n"
        f"Logged probability: {probabilities[-1]:.1%}\n"
        f"Samples: {sample_counts[-1]}",
        transform=rate_axis.transAxes,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_all_tasks(
    *,
    task_indices: list[int],
    task_labels: list[str],
    points_by_task: dict[int, list[tuple[int, float, int]]],
    probability_points_by_task: dict[int, list[tuple[int, float]]],
    run_last_step: int,
    sample_axis_max: int,
    output_path: Path,
) -> None:
    if len(task_indices) % 2 != 0:
        raise ValueError("The combined layout requires an even number of tasks")

    column_count = len(task_indices) // 2
    figure, axes = plt.subplots(
        2,
        column_count,
        figsize=(7 * column_count, 11),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    rate_color = "#62add8"
    probability_color = "#2a9d55"
    count_color = "#e07a1f"
    sample_locator = MaxNLocator(nbins=6, integer=True)
    sample_ticks = [
        tick
        for tick in sample_locator.tick_values(0, sample_axis_max)
        if 0 <= tick < sample_axis_max
    ]
    sample_ticks.append(sample_axis_max)

    # Pair consecutive tasks vertically: (0, 1), (2, 3), (4, 5), (6, 7).
    for column in range(column_count):
        for row_offset in range(2):
            list_index = 2 * column + row_offset
            task_index = task_indices[list_index]
            task_label = task_labels[list_index]
            rate_axis = axes[row_offset, column]
            count_axis = rate_axis.twinx()

            task_points = points_by_task[task_index]
            steps = [point[0] for point in task_points]
            success_rates = [point[1] for point in task_points]
            sample_counts = [point[2] for point in task_points]
            probability_points = probability_points_by_task[task_index]
            probability_steps = [point[0] for point in probability_points]
            probabilities = [point[1] for point in probability_points]

            plot_steps = list(steps)
            plot_rates = list(success_rates)
            plot_counts = list(sample_counts)
            if plot_steps[-1] < run_last_step:
                plot_steps.append(run_last_step)
                plot_rates.append(plot_rates[-1])
                plot_counts.append(plot_counts[-1])
            plot_probability_steps = list(probability_steps)
            plot_probabilities = list(probabilities)
            if plot_probability_steps[-1] < run_last_step:
                plot_probability_steps.append(run_last_step)
                plot_probabilities.append(plot_probabilities[-1])

            rate_axis.fill_between(
                plot_steps,
                plot_rates,
                step="post",
                color=rate_color,
                alpha=0.16,
                linewidth=0,
            )
            rate_axis.step(
                plot_steps,
                plot_rates,
                where="post",
                color=rate_color,
                linewidth=1.6,
            )
            rate_axis.scatter(
                steps,
                success_rates,
                color=rate_color,
                edgecolors="white",
                linewidths=0.3,
                s=9,
                zorder=3,
            )
            rate_axis.step(
                plot_probability_steps,
                plot_probabilities,
                where="post",
                color=probability_color,
                linewidth=1.4,
            )
            count_axis.step(
                plot_steps,
                plot_counts,
                where="post",
                color=count_color,
                linewidth=1.5,
            )

            rate_axis.set_xlim(0, run_last_step)
            rate_axis.set_ylim(0, 1)
            count_axis.set_ylim(0, sample_axis_max)
            count_axis.set_yticks(sample_ticks)
            rate_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            rate_axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
            count_axis.tick_params(axis="y", colors=count_color, labelsize=8)
            rate_axis.tick_params(axis="both", labelsize=8)
            rate_axis.grid(
                axis="both", color="#b0b0b0", alpha=0.28, linewidth=0.7
            )
            rate_axis.set_axisbelow(True)
            wrapped_label = textwrap.fill(task_label, width=32)
            rate_axis.set_title(
                f"Task {task_index}: {wrapped_label}\n"
                f"Final SR {success_rates[-1]:.0%} | "
                f"P {probabilities[-1]:.1%} | N {sample_counts[-1]}",
                fontsize=11,
                pad=9,
            )
            if column == 0:
                rate_axis.set_ylabel("Success rate / probability")
            if column == column_count - 1:
                count_axis.set_ylabel("Cumulative task samples", color=count_color)
            if row_offset == 1:
                rate_axis.set_xlabel("Global training step")

    legend_handles = [
        Patch(facecolor=rate_color, edgecolor=rate_color, alpha=0.16),
        Line2D([0], [0], color=probability_color, linewidth=1.8),
        Line2D([0], [0], color=count_color, linewidth=1.8),
    ]
    figure.legend(
        legend_handles,
        [
            "Rolling success rate",
            "Logged task probability (all-task normalized)",
            "Cumulative samples",
        ],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge local W&B sessions and generate one global-step versus "
            "success-rate/cumulative-sample figure per task."
        )
    )
    parser.add_argument("--wandb-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-label",
        action="append",
        dest="task_labels",
        help="Task label in task-index order; repeat once per task.",
    )
    args = parser.parse_args()

    session_dirs = sorted(args.wandb_root.glob(f"run-*-{args.run_id}"))
    wandb_files = [
        session_dir / f"run-{args.run_id}.wandb"
        for session_dir in session_dirs
        if (session_dir / f"run-{args.run_id}.wandb").is_file()
    ]
    if not wandb_files:
        raise FileNotFoundError(
            f"No local W&B files found for run ID {args.run_id!r} "
            f"under {args.wandb_root}"
        )

    points_by_task, probability_points_by_task, run_last_step = (
        _read_task_points(wandb_files)
    )
    if not points_by_task:
        raise RuntimeError(
            f"Run {args.run_id!r} has no per-task success-rate history"
        )
    task_indices = sorted(points_by_task)
    expected_indices = list(range(len(task_indices)))
    if task_indices != expected_indices:
        raise RuntimeError(
            f"Expected contiguous task indices {expected_indices}, got {task_indices}"
        )
    if sorted(probability_points_by_task) != task_indices:
        raise RuntimeError(
            "Per-task probability history does not match success-rate tasks: "
            f"{sorted(probability_points_by_task)} versus {task_indices}"
        )

    task_labels = args.task_labels or _infer_task_labels(
        session_dirs, len(task_indices)
    )
    if len(task_labels) != len(task_indices):
        raise ValueError(
            f"Got {len(task_labels)} labels for {len(task_indices)} tasks"
        )

    sample_axis_max = max(
        points[-1][2] for points in points_by_task.values()
    )

    for task_index, task_label in zip(task_indices, task_labels):
        points = points_by_task[task_index]
        output_path = args.output_dir / f"task_{task_index}_success_samples.png"
        _plot_task(
            task_index=task_index,
            task_label=task_label,
            points=points,
            probability_points=probability_points_by_task[task_index],
            run_last_step=run_last_step,
            sample_axis_max=sample_axis_max,
            output_path=output_path,
        )
        print(
            f"Task {task_index}: {points[-1][2]} cumulative samples, "
            f"final success rate {points[-1][1]:.1%} -> {output_path}"
        )

    combined_output = args.output_dir / "all_tasks_success_samples.png"
    _plot_all_tasks(
        task_indices=task_indices,
        task_labels=task_labels,
        points_by_task=points_by_task,
        probability_points_by_task=probability_points_by_task,
        run_last_step=run_last_step,
        sample_axis_max=sample_axis_max,
        output_path=combined_output,
    )
    print(f"Saved combined 2x{len(task_indices) // 2} figure to {combined_output}")

    print(
        f"Read {len(wandb_files)} W&B session files through global step "
        f"{run_last_step}; wrote {len(task_indices)} figures to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
