#!/usr/bin/env python3
"""Create a paper-style overview of per-task learning and sampling."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_task_success_samples import (
    _infer_task_labels,
    _read_task_points,
)


TEXT_COLOR = "#27333d"
SUCCESS_COLOR = "#2878aa"
PROBABILITY_COLOR = "#c66f0a"
SAMPLE_COLOR = "#6f63a8"
GRID_COLOR = "#d8dee3"
SPINE_COLOR = "#84919c"


def _style_axis(axis, *, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)
    axis.tick_params(colors=TEXT_COLOR, labelsize=9)
    if grid:
        axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)


def _extend_to_run_end(
    x: list[float], y: list[float], run_end_k: float
) -> tuple[list[float], list[float]]:
    plot_x = list(x)
    plot_y = list(y)
    if plot_x[-1] < run_end_k:
        plot_x.append(run_end_k)
        plot_y.append(plot_y[-1])
    return plot_x, plot_y


def _initial_probabilities(
    points_by_task: dict[int, list[tuple[int, float, int]]],
) -> dict[int, float]:
    initial_rates = {}
    for task_index, points in points_by_task.items():
        if points[0][0] != 0:
            raise RuntimeError(
                f"Task {task_index} has no step-0 success-rate baseline"
            )
        initial_rates[task_index] = points[0][1]
    weights = {
        task_index: max(1e-6, 1.0 - success_rate)
        for task_index, success_rate in initial_rates.items()
    }
    total = sum(weights.values())
    return {task_index: weight / total for task_index, weight in weights.items()}


def _probability_series(
    task_index: int,
    probability_points_by_task: dict[int, list[tuple[int, float]]],
    initial_probabilities: dict[int, float],
) -> tuple[list[float], list[float]]:
    points = probability_points_by_task[task_index]
    step_k = [0.0, *[step / 1_000 for step, _value in points]]
    probabilities = [
        100.0 * initial_probabilities[task_index],
        *[100.0 * value for _step, value in points],
    ]
    return step_k, probabilities


def _macro_success_series(
    points_by_task: dict[int, list[tuple[int, float, int]]],
) -> tuple[list[float], list[float]]:
    events: dict[int, dict[int, float]] = {}
    for task_index, points in points_by_task.items():
        for step, success_rate, _sample_count in points:
            events.setdefault(step, {})[task_index] = success_rate

    current_rates: dict[int, float] = {}
    step_k: list[float] = []
    macro_rates: list[float] = []
    for step in sorted(events):
        current_rates.update(events[step])
        if len(current_rates) != len(points_by_task):
            continue
        step_k.append(step / 1_000)
        macro_rates.append(100.0 * sum(current_rates.values()) / len(current_rates))
    return step_k, macro_rates


def _probability_matrix(
    task_indices: list[int],
    probability_points_by_task: dict[int, list[tuple[int, float]]],
    initial_probabilities: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    reference_steps = [
        step for step, _value in probability_points_by_task[task_indices[0]]
    ]
    rows = []
    for task_index in task_indices:
        task_points = probability_points_by_task[task_index]
        task_steps = [step for step, _value in task_points]
        if task_steps != reference_steps:
            raise RuntimeError(
                "Per-task probability histories are not aligned across tasks"
            )
        rows.append(
            [
                100.0 * initial_probabilities[task_index],
                *[100.0 * value for _step, value in task_points],
            ]
        )
    steps_k = np.asarray([0.0, *[step / 1_000 for step in reference_steps]])
    return steps_k, np.asarray(rows)


def _plot_summary(
    *,
    run_id: str,
    task_indices: list[int],
    task_labels: list[str],
    points_by_task: dict[int, list[tuple[int, float, int]]],
    probability_points_by_task: dict[int, list[tuple[int, float]]],
    run_last_step: int,
    output_path: Path,
    pdf_path: Path | None,
) -> None:
    if len(task_indices) != 8:
        raise ValueError(
            f"This paper layout expects exactly 8 tasks, got {len(task_indices)}"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )
    run_end_k = run_last_step / 1_000
    initial_probabilities = _initial_probabilities(points_by_task)
    sample_axis_max = max(
        points[-1][2] for points in points_by_task.values()
    )

    figure = plt.figure(figsize=(16.5, 9.8), facecolor="white")
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.03, 3.08),
        height_ratios=(1, 1),
        left=0.065,
        right=0.95,
        top=0.89,
        bottom=0.15,
        wspace=0.21,
        hspace=0.52,
    )

    # (a) Overall macro success.
    overall_axis = figure.add_subplot(outer[0, 0])
    macro_steps_k, macro_success = _macro_success_series(points_by_task)
    macro_steps_k, macro_success = _extend_to_run_end(
        macro_steps_k, macro_success, run_end_k
    )
    overall_axis.plot(
        macro_steps_k,
        macro_success,
        color=SUCCESS_COLOR,
        linewidth=3.0,
        drawstyle="steps-post",
    )
    overall_axis.set_xlim(0, run_end_k)
    overall_axis.set_ylim(0, 100)
    overall_axis.set_xticks([0, 100, 200])
    overall_axis.set_yticks([0, 50, 100])
    overall_axis.set_xlabel("Training steps (k)", fontsize=13)
    overall_axis.set_ylabel("Macro success (%)", fontsize=13)
    _style_axis(overall_axis)

    # (b) Probability allocation heatmap and horizontal colorbar.
    heatmap_grid = outer[1, 0].subgridspec(
        2, 1, height_ratios=(1, 0.08), hspace=0.42
    )
    heatmap_axis = figure.add_subplot(heatmap_grid[0, 0])
    colorbar_axis = figure.add_subplot(heatmap_grid[1, 0])
    probability_steps_k, probability_matrix = _probability_matrix(
        task_indices, probability_points_by_task, initial_probabilities
    )
    if probability_steps_k[-1] < run_end_k:
        x_edges = np.concatenate((probability_steps_k, [run_end_k]))
    else:
        median_width = float(np.median(np.diff(probability_steps_k)))
        x_edges = np.concatenate(
            (probability_steps_k, [probability_steps_k[-1] + median_width])
        )
    mesh = heatmap_axis.pcolormesh(
        x_edges,
        np.arange(len(task_indices) + 1),
        probability_matrix,
        cmap="YlOrBr",
        vmin=0,
        vmax=35,
        shading="flat",
        rasterized=True,
    )
    for separator in (2, 4, 6):
        heatmap_axis.axhline(separator, color="white", linewidth=2.0)
    heatmap_axis.set_xlim(0, run_end_k)
    heatmap_axis.set_xticks([0, 100, 200])
    heatmap_axis.set_yticks(
        np.arange(len(task_indices)) + 0.5,
        [f"T{index}" for index in task_indices],
    )
    heatmap_axis.invert_yaxis()
    heatmap_axis.set_xlabel("Training steps (k)", fontsize=13)
    heatmap_axis.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in heatmap_axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0, 12.5, 25, 35])
    colorbar.set_label("Sampling probability (%)", fontsize=12)
    colorbar.outline.set_visible(False)
    colorbar_axis.tick_params(labelsize=9, colors=TEXT_COLOR)

    # (c) Two task rows; each task has a success panel over a probability panel.
    per_task_grid = outer[:, 1].subgridspec(
        5,
        4,
        height_ratios=(1.0, 0.48, 0.42, 1.0, 0.48),
        wspace=0.23,
        hspace=0.14,
    )
    display_order = ((0, 2, 4, 6), (1, 3, 5, 7))
    for task_group_row, row_tasks in enumerate(display_order):
        success_row = 0 if task_group_row == 0 else 3
        probability_row = success_row + 1
        for column, task_index in enumerate(row_tasks):
            success_axis = figure.add_subplot(
                per_task_grid[success_row, column]
            )
            probability_axis = figure.add_subplot(
                per_task_grid[probability_row, column],
                sharex=success_axis,
            )
            sample_axis = probability_axis.twinx()

            task_points = points_by_task[task_index]
            success_steps_k = [step / 1_000 for step, _rate, _count in task_points]
            success_values = [100.0 * rate for _step, rate, _count in task_points]
            sample_steps_k = [step / 1_000 for step, _rate, _count in task_points]
            sample_values = [count for _step, _rate, count in task_points]
            success_steps_k, success_values = _extend_to_run_end(
                success_steps_k, success_values, run_end_k
            )
            sample_steps_k, sample_values = _extend_to_run_end(
                sample_steps_k, sample_values, run_end_k
            )
            probability_steps, probability_values = _probability_series(
                task_index,
                probability_points_by_task,
                initial_probabilities,
            )
            probability_steps, probability_values = _extend_to_run_end(
                probability_steps, probability_values, run_end_k
            )

            success_axis.plot(
                success_steps_k,
                success_values,
                color=SUCCESS_COLOR,
                linewidth=2.2,
                drawstyle="steps-post",
            )
            probability_axis.fill_between(
                probability_steps,
                probability_values,
                step="post",
                color=PROBABILITY_COLOR,
                alpha=0.12,
                linewidth=0,
            )
            probability_axis.plot(
                probability_steps,
                probability_values,
                color=PROBABILITY_COLOR,
                linewidth=1.9,
                drawstyle="steps-post",
            )
            probability_axis.axhline(
                12.5,
                color=SPINE_COLOR,
                linestyle=(0, (2, 2)),
                linewidth=1.0,
            )
            sample_axis.plot(
                sample_steps_k,
                sample_values,
                color=SAMPLE_COLOR,
                linewidth=1.5,
                drawstyle="steps-post",
            )

            success_axis.set_xlim(0, run_end_k)
            success_axis.set_ylim(0, 100)
            success_axis.set_yticks([0, 50, 100])
            probability_axis.set_ylim(0, 35)
            probability_axis.set_yticks([0, 30])
            sample_axis.set_ylim(0, sample_axis_max)
            sample_axis.set_yticks([0, 100, 200, sample_axis_max])
            probability_axis.set_xticks([0, 100, 200])
            _style_axis(success_axis)
            _style_axis(probability_axis)
            sample_axis.spines["top"].set_visible(False)
            sample_axis.spines["left"].set_visible(False)
            sample_axis.spines["right"].set_color(SPINE_COLOR)
            sample_axis.spines["bottom"].set_visible(False)
            sample_axis.tick_params(
                axis="y", colors=SAMPLE_COLOR, labelsize=7
            )
            success_axis.tick_params(labelbottom=False)
            if task_group_row == 0:
                probability_axis.tick_params(labelbottom=False)
            if column != 0:
                success_axis.tick_params(labelleft=False)
                probability_axis.tick_params(labelleft=False)
            else:
                success_axis.set_ylabel(
                    "Success (%)", color=SUCCESS_COLOR, fontsize=12
                )
                probability_axis.set_ylabel(
                    "Prob. (%)", color=PROBABILITY_COLOR, fontsize=11
                )
            if column != 3:
                sample_axis.tick_params(labelright=False, right=False)
                sample_axis.spines["right"].set_visible(False)
            else:
                sample_axis.set_ylabel(
                    "Cumulative samples", color=SAMPLE_COLOR, fontsize=9
                )

            full_title = textwrap.fill(
                f"T{task_index}  {task_labels[task_index]}", width=30
            )
            success_axis.set_title(
                full_title, fontsize=11.0, pad=9
            )

    # Section titles, shared x label, legend, and provenance note.
    figure.text(
        0.065,
        0.945,
        "(a) Overall performance",
        fontsize=16,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.065,
        0.515,
        "(b) Allocation across tasks",
        fontsize=16,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.385,
        0.945,
        "(c) Per-task learning and sampling",
        fontsize=16,
        fontweight="bold",
        ha="left",
    )
    figure.text(
        0.72,
        0.11,
        "Global training steps (k)",
        fontsize=13,
        ha="center",
    )
    legend_handles = [
        Line2D([0], [0], color=SUCCESS_COLOR, linewidth=3),
        Line2D([0], [0], color=PROBABILITY_COLOR, linewidth=2.5),
        Line2D([0], [0], color=SAMPLE_COLOR, linewidth=2.0),
        Line2D(
            [0],
            [0],
            color=SPINE_COLOR,
            linewidth=1.2,
            linestyle=(0, (2, 2)),
        ),
    ]
    figure.legend(
        legend_handles,
        [
            "Success rate",
            "Sampling probability",
            "Cumulative samples",
            "Uniform: 12.5%",
        ],
        loc="lower center",
        bbox_to_anchor=(0.72, 0.035),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        columnspacing=2.5,
    )
    figure.text(
        0.065,
        0.018,
        f"Actual W&B history from run {run_id}.",
        fontsize=10.5,
        color="#717d87",
    )
    figure.text(
        0.985,
        0.018,
        "Probability normalized over all 8 tasks; before feasibility filtering.",
        fontsize=10.5,
        color="#717d87",
        ha="right",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, facecolor="white")
    if pdf_path is not None:
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a paper-style task learning/sampling summary."
    )
    parser.add_argument("--wandb-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
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
    task_indices = sorted(points_by_task)
    if task_indices != list(range(8)):
        raise RuntimeError(f"Expected task indices 0--7, got {task_indices}")
    if sorted(probability_points_by_task) != task_indices:
        raise RuntimeError("Probability and success histories have different tasks")
    task_labels = _infer_task_labels(session_dirs, len(task_indices))

    _plot_summary(
        run_id=args.run_id,
        task_indices=task_indices,
        task_labels=task_labels,
        points_by_task=points_by_task,
        probability_points_by_task=probability_points_by_task,
        run_last_step=run_last_step,
        output_path=args.output,
        pdf_path=args.pdf,
    )
    print(
        f"Saved paper-style summary for 8 tasks through step {run_last_step} "
        f"to {args.output}"
    )
    if args.pdf is not None:
        print(f"Saved vector PDF to {args.pdf}")


if __name__ == "__main__":
    main()
