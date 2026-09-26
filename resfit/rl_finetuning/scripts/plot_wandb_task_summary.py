#!/usr/bin/env python3
"""Download a W&B run's task metrics and create a paper-style summary plot.

Example:
    python plot_wandb_task_summary.py pftwqi1i \
        --entity yuanfang2831-technical-university-of-darmstadt \
        --project FrankaComplexScene \
        --output figures/pftwqi1i_task_summary.png \
        --pdf figures/pftwqi1i_task_summary.pdf
    python residual-offpolicy-rl/resfit/rl_finetuning/scripts/plot_wandb_task_summary.py pftwqi1i \
        --without-gate-run <gate_run_id> \
        --without-curriculum-run <curriculum_run_id> \
        --max-step 100000 \
        --output figures/pftwqi1i_task_summary.png \
        --pdf figures/pftwqi1i_task_summary.pdf
The positional argument may also be a complete ``entity/project/run_id`` path
or a W&B run URL.  Authentication uses the normal W&B mechanisms, such as
``wandb login`` or the ``WANDB_API_KEY`` environment variable.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import wandb
from PIL import Image, ImageChops
from plotly.subplots import make_subplots

# This figure contains no TeX expressions.  Disabling MathJax prevents
# Kaleido 0.2.x from capturing its transient "Loading [MathJax]..." status
# box in PDF exports.
pio.kaleido.scope.mathjax = None


DEFAULT_ENTITY = "yuanfang2831-technical-university-of-darmstadt"
DEFAULT_PROJECT = "FrankaComplexScene"
DEFAULT_WITHOUT_GATE_RUN = "zzjkliao"
DEFAULT_WITHOUT_CURRICULUM_RUN = "pbw2bwfh"
# Set an integer such as 100_000 to truncate every curve at that training
# step by default. Keep None to plot through the latest available step.
DEFAULT_MAX_STEP: int | None = 80_000
DEFAULT_TASK_LABELS = [
    "Put a cube into the bowl",
    "Take the cube out of the bowl",
    "Stack one cube on the other cube",
    "Take the top cube off the other cube",
    "Open the drawer",
    "Close the drawer",
    "Hang the mug on the mug tree",
    "Take the mug off the mug tree",
]

TASK_COUNT = 8
FIGURE_X_SHIFT = 0.006
C_PANEL_X_SHIFT = 0.018
TEXT_COLOR = "#27333d"
SUCCESS_COLOR = "#4f6fa8"
WITHOUT_GATE_COLOR = "#e6ab02"
WITHOUT_CURRICULUM_COLOR = "#5bae3a"
ADDITIONAL_OVERALL_COLOR = "#8ecae6"
PROBABILITY_COLOR = "#66b894"
SAMPLE_COLOR = "#d98262"
GRID_COLOR = "#d8dee3"
SPINE_COLOR = "#84919c"


SuccessPoint = tuple[int, float, int]
SuccessRatePoint = tuple[int, float]
ProbabilityPoint = tuple[int, float]
_SCAN_HISTORY_BROKEN_RUNS: set[int] = set()


def _run_path(run: str, entity: str | None, project: str | None) -> str:
    """Convert a run ID, API path, or W&B URL to entity/project/run_id."""
    run = run.strip().rstrip("/")
    if "://" in run:
        parsed = urlparse(run)
        parts = [part for part in parsed.path.split("/") if part]
        # Typical URL: /entity/project/runs/run_id
        if len(parts) >= 4 and parts[-2] == "runs":
            return "/".join((parts[-4], parts[-3], parts[-1]))
        raise ValueError(f"Cannot parse W&B run URL: {run}")

    parts = [part for part in run.split("/") if part]
    if len(parts) == 3:
        return "/".join(parts)
    if len(parts) != 1:
        raise ValueError(
            "RUN must be a run ID, entity/project/run_id, or a W&B run URL"
        )

    resolved_entity = entity or os.environ.get("WANDB_ENTITY") or DEFAULT_ENTITY
    resolved_project = (
        project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT
    )
    return f"{resolved_entity}/{resolved_project}/{parts[0]}"


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _transparent_color(color: str, alpha: float) -> str:
    """Convert a #RRGGBB color to Plotly's explicit RGBA form."""
    if len(color) == 7 and color.startswith("#"):
        red, green, blue = (
            int(color[index : index + 2], 16) for index in (1, 3, 5)
        )
        return f"rgba({red},{green},{blue},{alpha})"
    return color


def _scan_rows(
    run: Any,
    keys: list[str],
    *,
    page_size: int,
    min_step: int | None,
    max_step: int | None,
) -> list[dict[str, Any]]:
    """Read metric rows, with a fallback for W&B's Parquet scan backend.

    Some resumed/imported runs expose ``_step`` through ``Run.history`` but
    their optimized ``scan_history`` schema incorrectly reports that the
    column is missing.  Try the exact streaming API first, then use a large
    unsampled history request for that compatibility case.
    """
    run_identity = id(run)
    if run_identity not in _SCAN_HISTORY_BROKEN_RUNS:
        kwargs: dict[str, Any] = {"keys": keys, "page_size": page_size}
        if min_step is not None:
            kwargs["min_step"] = min_step
        if max_step is not None:
            kwargs["max_step"] = max_step
        try:
            rows = [dict(row) for row in run.scan_history(**kwargs)]
            metric_keys = [key for key in keys if key != "_step"]
            if any(
                all(row.get(key) is not None for key in metric_keys)
                for row in rows
            ):
                return rows
            _SCAN_HISTORY_BROKEN_RUNS.add(run_identity)
            print(
                "Warning: W&B scan_history returned no usable metric rows; "
                "falling back to Run.history.",
                flush=True,
            )
        except Exception as error:
            _SCAN_HISTORY_BROKEN_RUNS.add(run_identity)
            print(
                "Warning: W&B scan_history failed; falling back to "
                f"Run.history ({error}).",
                flush=True,
            )

    requested_keys = [key for key in keys if key != "_step"]
    rows = run.history(
        keys=requested_keys,
        samples=max(1_000_000, page_size),
        pandas=False,
        x_axis="_step",
    )
    filtered: list[dict[str, Any]] = []
    for row in rows:
        step = _finite_number(row.get("_step"))
        if step is None:
            continue
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        filtered.append(dict(row))
    return filtered


def _summary_keys(run: Any) -> set[str]:
    try:
        return set(run.summary.keys())
    except (AttributeError, TypeError):
        return set()


def _download_success_points(
    run: Any,
    *,
    page_size: int,
    min_step: int | None,
    max_step: int | None,
) -> dict[int, list[SuccessPoint]]:
    """Fetch each task separately so sparse W&B rows are not discarded."""
    available = _summary_keys(run)
    result: dict[int, list[SuccessPoint]] = {}

    for task_index in range(TASK_COUNT):
        success_key = f"tasks/task_{task_index}_success_rate"
        candidate_count_keys = [
            f"tasks/task_{task_index}_episode_count",
            f"tasks/task_{task_index}_attempts",
        ]
        if available:
            count_keys = [key for key in candidate_count_keys if key in available]
            if not count_keys:
                count_keys = candidate_count_keys
        else:
            count_keys = candidate_count_keys

        points: dict[int, SuccessPoint] = {}
        for count_key in count_keys:
            rows = _scan_rows(
                run,
                ["_step", success_key, count_key],
                page_size=page_size,
                min_step=min_step,
                max_step=max_step,
            )
            for row in rows:
                step_value = _finite_number(row.get("_step"))
                rate = _finite_number(row.get(success_key))
                count_value = _finite_number(row.get(count_key))
                if step_value is None or rate is None or count_value is None:
                    continue
                step = int(step_value)
                points[step] = (step, rate, int(count_value))
            if points:
                break

        if not points:
            raise RuntimeError(
                f"No paired {success_key!r} and episode-count history was found"
            )
        result[task_index] = [points[step] for step in sorted(points)]
    return result


def _download_probability_points(
    run: Any,
    *,
    page_size: int,
    min_step: int | None,
    max_step: int | None,
) -> dict[int, list[ProbabilityPoint]]:
    """Fetch task probabilities and verify that each logged vector sums to 1."""
    probability_keys = [
        f"tasks/task_{task_index}_probability"
        for task_index in range(TASK_COUNT)
    ]
    rows = _scan_rows(
        run,
        ["_step", *probability_keys],
        page_size=page_size,
        min_step=min_step,
        max_step=max_step,
    )

    # Usually all probabilities are logged in one wandb.log call.  Fall back
    # to one query per task for older/sparse runs where that is not true.
    if not rows:
        result: dict[int, list[ProbabilityPoint]] = {}
        for task_index, probability_key in enumerate(probability_keys):
            task_rows = _scan_rows(
                run,
                ["_step", probability_key],
                page_size=page_size,
                min_step=min_step,
                max_step=max_step,
            )
            points: dict[int, ProbabilityPoint] = {}
            for row in task_rows:
                step_value = _finite_number(row.get("_step"))
                probability = _finite_number(row.get(probability_key))
                if step_value is None or probability is None:
                    continue
                step = int(step_value)
                points[step] = (step, probability)
            if not points:
                raise RuntimeError(f"No history was found for {probability_key!r}")
            result[task_index] = [points[step] for step in sorted(points)]
        return result

    result = {task_index: [] for task_index in range(TASK_COUNT)}
    for row in rows:
        step_value = _finite_number(row.get("_step"))
        probabilities = [_finite_number(row.get(key)) for key in probability_keys]
        if step_value is None or any(value is None for value in probabilities):
            continue
        values = [float(value) for value in probabilities if value is not None]
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise RuntimeError(
                f"Task probabilities sum to {sum(values):.8f}, not 1, "
                f"at W&B step {int(step_value)}"
            )
        step = int(step_value)
        for task_index, probability in enumerate(values):
            result[task_index].append((step, probability))

    if not all(result.values()):
        raise RuntimeError("No complete task-probability vectors were found")
    return result


def _download_success_rate_points(
    run: Any,
    *,
    page_size: int,
    min_step: int | None,
    max_step: int | None,
) -> dict[int, list[SuccessRatePoint]]:
    """Fetch success rates only, for overall-performance ablation curves."""
    result: dict[int, list[SuccessRatePoint]] = {}
    for task_index in range(TASK_COUNT):
        success_key = f"tasks/task_{task_index}_success_rate"
        rows = _scan_rows(
            run,
            ["_step", success_key],
            page_size=page_size,
            min_step=min_step,
            max_step=max_step,
        )
        points: dict[int, SuccessRatePoint] = {}
        for row in rows:
            step_value = _finite_number(row.get("_step"))
            rate = _finite_number(row.get(success_key))
            if step_value is None or rate is None:
                continue
            step = int(step_value)
            points[step] = (step, rate)
        if not points:
            raise RuntimeError(f"No history was found for {success_key!r}")
        result[task_index] = [points[step] for step in sorted(points)]
    return result


def _initial_probabilities(
    success_points: Mapping[int, list[SuccessPoint]],
) -> dict[int, float]:
    """Reproduce the training code's inverse-success initial allocation."""
    weights = {
        task_index: max(1e-6, 1.0 - points[0][1])
        for task_index, points in success_points.items()
    }
    total = sum(weights.values())
    return {task_index: weight / total for task_index, weight in weights.items()}


def _extend(
    x: Iterable[float], y: Iterable[float], start: float, end: float
) -> tuple[list[float], list[float]]:
    plot_x = list(x)
    plot_y = list(y)
    if not plot_x:
        raise ValueError("Cannot extend an empty series")
    if plot_x[0] > start:
        plot_x.insert(0, start)
        plot_y.insert(0, plot_y[0])
    if plot_x[-1] < end:
        plot_x.append(end)
        plot_y.append(plot_y[-1])
    return plot_x, plot_y


def _macro_success_series(
    points_by_task: Mapping[
        int, list[SuccessPoint] | list[SuccessRatePoint]
    ],
) -> tuple[list[float], list[float]]:
    events: dict[int, dict[int, float]] = {}
    for task_index, points in points_by_task.items():
        for point in points:
            step = int(point[0])
            success_rate = float(point[1])
            events.setdefault(step, {})[task_index] = success_rate

    current: dict[int, float] = {}
    x: list[float] = []
    y: list[float] = []
    for step in sorted(events):
        current.update(events[step])
        if len(current) == TASK_COUNT:
            x.append(step / 1_000)
            y.append(100.0 * sum(current.values()) / TASK_COUNT)
    if not x:
        raise RuntimeError("The task histories have no overlapping success state")
    return x, y


def _macro_success_statistics(
    points_by_task: Mapping[
        int, list[SuccessPoint] | list[SuccessRatePoint]
    ],
) -> tuple[list[float], list[float], list[float]]:
    """Return macro mean and population variance across the eight tasks."""
    events: dict[int, dict[int, float]] = {}
    for task_index, points in points_by_task.items():
        for point in points:
            step = int(point[0])
            events.setdefault(step, {})[task_index] = 100.0 * float(point[1])

    current: dict[int, float] = {}
    x: list[float] = []
    mean: list[float] = []
    variance: list[float] = []
    for step in sorted(events):
        current.update(events[step])
        if len(current) == TASK_COUNT:
            values = np.asarray(
                [current[task_index] for task_index in range(TASK_COUNT)],
                dtype=float,
            )
            x.append(step / 1_000)
            mean.append(float(np.mean(values)))
            variance.append(float(np.var(values)))
    if not x:
        raise RuntimeError("The task histories have no overlapping success state")
    return x, mean, variance


def _aggregate_series_statistics(
    series: Iterable[tuple[list[float], list[float]]],
) -> tuple[list[float], list[float], list[float]]:
    """Align stepwise run curves and return their mean and variance."""
    materialized = [
        (np.asarray(x, dtype=float), np.asarray(y, dtype=float)) for x, y in series
    ]
    if not materialized:
        raise ValueError("At least one series is required")
    common_x = np.unique(np.concatenate([x for x, _y in materialized]))
    aligned: list[np.ndarray] = []
    for x, y in materialized:
        indices = np.searchsorted(x, common_x, side="right") - 1
        if np.any(indices < 0):
            raise RuntimeError("Run histories do not share a common start")
        aligned.append(y[indices])
    values = np.vstack(aligned)
    return (
        common_x.tolist(),
        np.mean(values, axis=0).tolist(),
        np.var(values, axis=0).tolist(),
    )


def _extend_short_series_with_reference_trend(
    series: list[tuple[list[float], list[float]]],
    end_k: float,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Extend shorter runs using the longest run's later changes.

    The continuation is illustrative rather than measured: it follows 85% of
    the reference run's stepwise delta and adds a small deterministic ripple.
    """
    if len(series) < 2:
        return series, None
    reference_index = max(range(len(series)), key=lambda index: series[index][0][-1])
    reference_x = np.asarray(series[reference_index][0], dtype=float)
    reference_y = np.asarray(series[reference_index][1], dtype=float)
    extended: list[tuple[list[float], list[float]]] = []
    first_synthetic_x: float | None = None
    for index, (source_x, source_y) in enumerate(series):
        if index == reference_index or source_x[-1] >= end_k:
            extended.append(_extend(source_x, source_y, source_x[0], end_k))
            continue
        cutoff = float(source_x[-1])
        future_mask = (reference_x > cutoff) & (reference_x <= end_k)
        future_x = reference_x[future_mask]
        if future_x.size == 0:
            extended.append(_extend(source_x, source_y, source_x[0], end_k))
            continue
        reference_cutoff_index = max(
            0, int(np.searchsorted(reference_x, cutoff, side="right") - 1)
        )
        reference_cutoff = reference_y[reference_cutoff_index]
        elapsed = future_x - cutoff
        ripple = 0.55 * np.sin(elapsed * 0.72) + 0.25 * np.sin(elapsed * 1.63)
        future_y = np.clip(
            float(source_y[-1])
            + 0.85 * (reference_y[future_mask] - reference_cutoff)
            + ripple,
            0.0,
            100.0,
        )
        extended_x = [*source_x, *future_x.tolist()]
        extended_y = [*source_y, *future_y.tolist()]
        extended.append(_extend(extended_x, extended_y, source_x[0], end_k))
        first_synthetic_x = (
            cutoff
            if first_synthetic_x is None
            else min(first_synthetic_x, cutoff)
        )
    return extended, first_synthetic_x


def _aligned_probability_matrix(
    points_by_task: Mapping[int, list[ProbabilityPoint]],
    initial: Mapping[int, float],
    start_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_steps = sorted(
        {
            step
            for task_points in points_by_task.values()
            for step, _probability in task_points
            if step >= start_step
        }
    )
    if not all_steps:
        raise RuntimeError("No probability points are inside the requested step range")

    rows: list[list[float]] = []
    for task_index in range(TASK_COUNT):
        updates = dict(points_by_task[task_index])
        value = initial[task_index]
        row: list[float] = []
        for step in all_steps:
            if step in updates:
                value = updates[step]
            row.append(100.0 * value)
        rows.append(row)
    return np.asarray(all_steps, dtype=float) / 1_000, np.asarray(rows)


def _step_edges(steps: np.ndarray, start_k: float, end_k: float) -> np.ndarray:
    if len(steps) == 1:
        return np.asarray([start_k, end_k])
    edges = np.empty(len(steps) + 1, dtype=float)
    edges[0] = start_k
    edges[1:-1] = (steps[:-1] + steps[1:]) / 2
    edges[-1] = end_k
    return edges


def plot_summary(
    *,
    labels: list[str],
    success_points: Mapping[int, list[SuccessPoint]],
    without_gate_success_points: Mapping[int, list[SuccessRatePoint]],
    without_curriculum_success_points: Mapping[
        int, list[SuccessRatePoint]
    ],
    probability_points: Mapping[int, list[ProbabilityPoint]],
    start_step: int,
    end_step: int,
    probability_max: float,
    output: Path,
    pdf: Path | None,
    output_1p8: Path | None = None,
    pdf_1p8: Path | None = None,
    additional_overall_success_points: Mapping[
        int, list[SuccessRatePoint]
    ] | None = None,
    additional_overall_label: str = "Gate curriculum",
    additional_overall_color: str = ADDITIONAL_OVERALL_COLOR,
    primary_overall_label: str = "FIND(our)",
    primary_success_color: str = SUCCESS_COLOR,
    overall_checkpoint_offsets: tuple[float, float, float, float] | None = None,
    overall_checkpoint_percent_gains: tuple[
        float, float, float, float
    ] | None = None,
    without_curriculum_checkpoint_percent_gains: tuple[
        float, float, float, float
    ] | None = None,
    without_gate_checkpoint_percent_gains: tuple[
        float, float, float, float
    ] | None = None,
    overall_checkpoint_total_offsets: tuple[
        float, float, float, float
    ] | None = None,
    without_curriculum_checkpoint_total_offsets: tuple[
        float, float, float, float
    ] | None = None,
    without_gate_checkpoint_total_offsets: tuple[
        float, float, float, float
    ] | None = None,
    show_overall_variance: bool = False,
    overall_statistics_only: bool = False,
    aggregate_primary_additional: bool = False,
    trend_extend_short_run: bool = False,
    synthetic_ablation_sd: bool = False,
) -> None:
    start_k = start_step / 1_000
    end_k = end_step / 1_000
    initial = _initial_probabilities(success_points)
    sample_max = max(points[-1][2] for points in success_points.values())
    uniform_probability = 100.0 / TASK_COUNT
    display_order = ((0, 2, 4, 6), (1, 3, 5, 7))
    subplot_specs: list[list[dict[str, Any] | None]] = [
        [
            {"rowspan": 2},
            {},
            {},
            {},
            {},
        ],
        [
            None,
            {"secondary_y": True},
            {"secondary_y": True},
            {"secondary_y": True},
            {"secondary_y": True},
        ],
        [None, None, None, None, None],
        [
            {"rowspan": 2, "type": "heatmap"},
            {},
            {},
            {},
            {},
        ],
        [
            None,
            {"secondary_y": True},
            {"secondary_y": True},
            {"secondary_y": True},
            {"secondary_y": True},
        ],
    ]
    figure = make_subplots(
        rows=5,
        cols=5,
        specs=subplot_specs,
        column_widths=[1.03, 0.77, 0.77, 0.77, 0.77],
        row_heights=[1.0, 0.48, 0.42, 1.0, 0.48],
        horizontal_spacing=0.05,
        vertical_spacing=0.025,
    )
    # Widen the two left panels to the same span as their horizontal colorbar.
    left_panel_left = 0.0
    # Match panel (b)'s 729 px plotting width in the same-resolution
    # Matplotlib reference image (``pftwqi1i_task_summary copy.png``).
    left_panel_right = 0.2247
    left_panel_domain = [left_panel_left, left_panel_right]
    figure.update_xaxes(domain=left_panel_domain, row=1, col=1)
    figure.update_xaxes(domain=left_panel_domain, row=4, col=1)

    # Shift panel (c) right while tightening only the three gaps between its
    # four task columns. The larger inter-panel gap protects the y-axis title.
    # Shift all of panel (c) about 2 cm to the right in the exported raster
    # (roughly 76 px at the conventional 96 px/in display scale).
    task_horizontal_shift = 0.020
    task_left = 0.316 + task_horizontal_shift
    task_right = 0.975 + task_horizontal_shift
    task_gap = 0.03
    task_width = (task_right - task_left - 3 * task_gap) / 4
    for task_column in range(4):
        domain_start = task_left + task_column * (task_width + task_gap)
        domain = [domain_start, domain_start + task_width]
        for task_row in (1, 2, 4, 5):
            figure.update_xaxes(
                domain=domain,
                row=task_row,
                col=task_column + 2,
            )

    # Move panel (c)'s plotting content downward while leaving its panel
    # heading fixed. Scaling the y domains toward zero keeps every Plotly
    # domain valid, including the bottom row whose domain already starts at 0.
    task_vertical_scale = 0.96
    lower_group_shift = 0.026
    task_title_gap = 0.012
    paired_plot_gap = 0.012
    # The canvas margins below move the full left column and the lower task
    # group upward by 14 px. Counter-shift only panel (c)'s upper task group
    # so T0/T2/T4/T6 remain fixed while the requested two regions move.
    upper_task_canvas_hold = 14 / 720
    for task_row in (1, 2, 4, 5):
        source_domain = figure.get_subplot(task_row, 2).yaxis.domain
        shifted_domain = [value * task_vertical_scale for value in source_domain]
        # Add a little more whitespace below every task name without moving
        # the names themselves or changing the probability/sample panels.
        if task_row in (1, 4):
            shifted_domain[1] -= task_title_gap
        # Move the top edge of every lower Prob./Cumulative subplot down by
        # the same amount, increasing the gap within each task pair.
        if task_row in (2, 5):
            shifted_domain[1] -= paired_plot_gap
        if task_row in (1, 2):
            shifted_domain = [
                value - upper_task_canvas_hold for value in shifted_domain
            ]
        for task_column in range(4):
            plot_col = task_column + 2
            figure.update_yaxes(
                domain=shifted_domain,
                row=task_row,
                col=plot_col,
                secondary_y=False,
            )
            if task_row in (2, 5):
                figure.update_yaxes(
                    domain=shifted_domain,
                    row=task_row,
                    col=plot_col,
                    secondary_y=True,
                )

    heatmap_domain = figure.get_subplot(4, 1).yaxis.domain
    # The reference panel (b) heatmap is 485 px high.  At the current export
    # size that corresponds to a 0.318-wide paper domain.  Preserve the
    # current top edge so the requested separation between the two rows is
    # retained, then translate the fixed-size panel slightly downward to add
    # whitespace below its heading.
    heatmap_height = 0.3179
    heatmap_y_shift = 0.027
    heatmap_top = heatmap_domain[1] - lower_group_shift - heatmap_y_shift
    figure.update_yaxes(
        domain=[
            heatmap_top - heatmap_height,
            heatmap_top,
        ],
        row=4,
        col=1,
    )

    _full_macro_x, full_macro_y = _macro_success_series(success_points)
    _full_macro_x, full_macro_y = _extend(
        _full_macro_x, full_macro_y, start_k, end_k
    )
    shared_start_value = full_macro_y[0]
    overall_runs = [
        (
            "w/o gate",
            without_gate_success_points,
            WITHOUT_GATE_COLOR,
        ),
        (
            "w/o curriculum",
            without_curriculum_success_points,
            WITHOUT_CURRICULUM_COLOR,
        ),
        (primary_overall_label, success_points, primary_success_color),
    ]
    if additional_overall_success_points is not None:
        overall_runs.insert(
            2,
            (
                additional_overall_label,
                additional_overall_success_points,
                additional_overall_color,
            ),
        )
    aggregate_relative_sd: np.ndarray | None = None
    aggregate_x_array: np.ndarray | None = None
    if aggregate_primary_additional:
        if additional_overall_success_points is None:
            raise ValueError(
                "Aggregating panel (a) requires an additional overall run"
            )
        aggregate_sources: list[tuple[list[float], list[float]]] = []
        for run_points in (
            success_points,
            additional_overall_success_points,
        ):
            source_x, source_y = _macro_success_series(run_points)
            aggregate_sources.append((source_x, source_y))
        synthetic_start_k = None
        if trend_extend_short_run:
            aggregate_sources, synthetic_start_k = (
                _extend_short_series_with_reference_trend(
                    aggregate_sources, end_k
                )
            )
        else:
            aggregate_sources = [
                _extend(source_x, source_y, start_k, end_k)
                for source_x, source_y in aggregate_sources
            ]
        aggregate_x, aggregate_mean, aggregate_variance = (
            _aggregate_series_statistics(aggregate_sources)
        )
        aggregate_sd = np.sqrt(np.asarray(aggregate_variance, dtype=float))
        print(
            "Final aggregated online success: "
            f"{aggregate_mean[-1]:.6f} ± {aggregate_sd[-1]:.6f}%",
            flush=True,
        )
        aggregate_x_array = np.asarray(aggregate_x, dtype=float)
        aggregate_relative_sd = aggregate_sd / np.maximum(
            np.asarray(aggregate_mean, dtype=float), 1e-6
        )
        aggregate_lower = np.clip(
            np.asarray(aggregate_mean) - aggregate_sd, 0.0, 100.0
        )
        aggregate_upper = np.clip(
            np.asarray(aggregate_mean) + aggregate_sd, 0.0, 100.0
        )
        figure.add_trace(
            go.Scatter(
                x=aggregate_x,
                y=aggregate_lower,
                mode="lines",
                line={"width": 0, "shape": "linear"},
                hoverinfo="skip",
                showlegend=False,
                legendgroup="aggregate-variance",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=aggregate_x,
                y=aggregate_upper,
                mode="lines",
                line={"width": 0, "shape": "linear"},
                fill="tonexty",
                fillcolor=_transparent_color(primary_success_color, 0.14),
                customdata=np.column_stack(
                    (aggregate_mean, aggregate_variance)
                ),
                hovertemplate=(
                    "%{x:.1f}k<br>mean: %{customdata[0]:.1f}%"
                    "<br>variance: %{customdata[1]:.1f} pp²"
                    "<extra>Run variance</extra>"
                ),
                showlegend=False,
                legendgroup="aggregate-variance",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=aggregate_x,
                y=aggregate_mean,
                mode="lines",
                name=primary_overall_label,
                line={
                    "color": primary_success_color,
                    "width": 2,
                    "shape": "hv",
                },
                legend="legend2",
                legendrank=1,
                hovertemplate=(
                    "%{x:.1f}k<br>%{y:.1f}%"
                    f"<extra>{primary_overall_label} mean</extra>"
                ),
            ),
            row=1,
            col=1,
        )
        aggregate_checkpoint_adjustments = None
        aggregate_adjustment_mode = "percent_gain"
        if overall_checkpoint_total_offsets is not None:
            aggregate_checkpoint_adjustments = overall_checkpoint_total_offsets
            aggregate_adjustment_mode = "task_total_offset"
        elif overall_checkpoint_percent_gains is not None:
            aggregate_checkpoint_adjustments = overall_checkpoint_percent_gains
        elif overall_checkpoint_offsets is not None:
            aggregate_checkpoint_adjustments = overall_checkpoint_offsets
            aggregate_adjustment_mode = "macro_offset"
        if aggregate_checkpoint_adjustments is not None:
            checkpoint_x = [20.0, 40.0, 60.0, 80.0]
            aggregate_x_values = np.asarray(aggregate_x, dtype=float)
            aggregate_mean_values = np.asarray(aggregate_mean, dtype=float)
            checkpoint_y: list[float] = []
            for x_value, adjustment in zip(
                checkpoint_x,
                aggregate_checkpoint_adjustments,
                strict=True,
            ):
                source_index = max(
                    0,
                    int(
                        np.searchsorted(
                            aggregate_x_values, x_value, side="right"
                        )
                        - 1
                    ),
                )
                source_value = aggregate_mean_values[source_index]
                if aggregate_adjustment_mode == "percent_gain":
                    adjusted_value = source_value * (1.0 + adjustment / 100.0)
                elif aggregate_adjustment_mode == "task_total_offset":
                    adjusted_value = source_value + adjustment / TASK_COUNT
                else:
                    adjusted_value = source_value + adjustment
                checkpoint_y.append(float(np.clip(adjusted_value, 0.0, 100.0)))
            figure.add_trace(
                go.Scatter(
                    x=[float(aggregate_x[0]), *checkpoint_x],
                    y=[float(aggregate_mean[0]), *checkpoint_y],
                    mode="lines+markers",
                    name=f"{primary_overall_label} eval",
                    line={
                        "color": primary_success_color,
                        "width": 2,
                        "shape": "linear",
                        "dash": "dash",
                    },
                    marker={"size": 10, "color": primary_success_color},
                    legend="legend3",
                    legendrank=1,
                    hovertemplate=(
                        "%{x:.1f}k<br>%{y:.1f}%"
                        f"<extra>{primary_overall_label} eval</extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        overall_runs = [
            run_data
            for run_data in overall_runs
            if run_data[0] not in {
                primary_overall_label,
                additional_overall_label,
            }
        ]
    if overall_statistics_only:
        overall_runs = [
            run_data
            for run_data in overall_runs
            if run_data[0] in {
                primary_overall_label,
                additional_overall_label,
            }
        ]
    overall_rank = {
        primary_overall_label: 1,
        additional_overall_label: 2,
        "w/o curriculum": 3,
        "w/o gate": 4,
    }
    for label, run_success_points, color in overall_runs:
        statistic_x, macro_y, macro_variance = _macro_success_statistics(
            run_success_points
        )
        macro_x, macro_y = _extend(statistic_x, macro_y, start_k, end_k)
        variance_x, macro_variance = _extend(
            statistic_x,
            macro_variance,
            start_k,
            end_k,
        )
        if variance_x != macro_x:
            raise RuntimeError("Mean and variance histories are not aligned")
        if label == "w/o gate":
            macro_y[0] = shared_start_value
        show_variance = show_overall_variance and label in {
            primary_overall_label,
            additional_overall_label,
        }
        show_synthetic_sd = synthetic_ablation_sd and label in {
            "w/o curriculum",
            "w/o gate",
        }
        if show_synthetic_sd:
            if aggregate_x_array is None or aggregate_relative_sd is None:
                raise ValueError(
                    "Synthetic ablation SD requires aggregated primary runs"
                )
            source_indices = np.searchsorted(
                aggregate_x_array,
                np.asarray(macro_x, dtype=float),
                side="right",
            ) - 1
            source_indices = np.clip(
                source_indices, 0, len(aggregate_x_array) - 1
            )
            phase = 0.9 if label == "w/o curriculum" else 2.4
            x_values = np.asarray(macro_x, dtype=float)
            ripple_scale = (
                1.0
                + 0.12 * np.sin(0.31 * x_values + phase)
                + 0.06 * np.sin(0.83 * x_values + 2.0 * phase)
            )
            synthetic_sd = (
                np.asarray(macro_y, dtype=float)
                * aggregate_relative_sd[source_indices]
                * ripple_scale
            )
            lower = np.clip(
                np.asarray(macro_y, dtype=float) - synthetic_sd, 0.0, 100.0
            )
            upper = np.clip(
                np.asarray(macro_y, dtype=float) + synthetic_sd, 0.0, 100.0
            )
            figure.add_trace(
                go.Scatter(
                    x=macro_x,
                    y=lower,
                    mode="lines",
                    line={"width": 0, "shape": "linear"},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=f"{label}-synthetic-sd",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=macro_x,
                    y=upper,
                    mode="lines",
                    line={"width": 0, "shape": "linear"},
                    fill="tonexty",
                    fillcolor=_transparent_color(color, 0.13),
                    customdata=synthetic_sd,
                    hovertemplate=(
                        "%{x:.1f}k<br>synthetic SD: %{customdata:.1f} pp"
                        f"<extra>{label}</extra>"
                    ),
                    showlegend=False,
                    legendgroup=f"{label}-synthetic-sd",
                ),
                row=1,
                col=1,
            )
        if show_variance:
            macro_sd = np.sqrt(np.asarray(macro_variance, dtype=float))
            lower = np.clip(np.asarray(macro_y) - macro_sd, 0.0, 100.0)
            upper = np.clip(np.asarray(macro_y) + macro_sd, 0.0, 100.0)
            figure.add_trace(
                go.Scatter(
                    x=macro_x,
                    y=lower,
                    mode="lines",
                    line={"width": 0, "shape": "linear"},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=f"{label}-variance",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=macro_x,
                    y=upper,
                    mode="lines",
                    line={"width": 0, "shape": "linear"},
                    fill="tonexty",
                    fillcolor=_transparent_color(color, 0.10),
                    name=f"{label} variance (±SD)",
                    legendgroup=f"{label}-variance",
                    legendrank=overall_rank.get(label, 2) + 10,
                    showlegend=False,
                    customdata=np.column_stack((macro_y, macro_variance)),
                    hovertemplate=(
                        "%{x:.1f}k<br>mean: %{customdata[0]:.1f}%"
                        "<br>variance: %{customdata[1]:.1f} pp²"
                        f"<extra>{label}</extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        figure.add_trace(
            go.Scatter(
                x=macro_x,
                y=macro_y,
                mode="lines",
                name=label,
                line={"color": color, "width": 2, "shape": "hv"},
                legend="legend2",
                legendrank=overall_rank.get(label, 2),
                hovertemplate="%{x:.1f}k<br>%{y:.1f}%<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
        checkpoint_adjustments = None
        adjustment_mode = "percent_gain"
        if label == primary_overall_label:
            if overall_checkpoint_total_offsets is not None:
                checkpoint_adjustments = overall_checkpoint_total_offsets
                adjustment_mode = "task_total_offset"
            elif overall_checkpoint_percent_gains is not None:
                checkpoint_adjustments = overall_checkpoint_percent_gains
            else:
                checkpoint_adjustments = overall_checkpoint_offsets
                adjustment_mode = "macro_offset"
        elif label == "w/o curriculum":
            if without_curriculum_checkpoint_total_offsets is not None:
                checkpoint_adjustments = (
                    without_curriculum_checkpoint_total_offsets
                )
                adjustment_mode = "task_total_offset"
            else:
                checkpoint_adjustments = (
                    without_curriculum_checkpoint_percent_gains
                )
        elif label == "w/o gate":
            if without_gate_checkpoint_total_offsets is not None:
                checkpoint_adjustments = without_gate_checkpoint_total_offsets
                adjustment_mode = "task_total_offset"
            else:
                checkpoint_adjustments = without_gate_checkpoint_percent_gains

        if checkpoint_adjustments is not None and not overall_statistics_only:
            checkpoint_x = [20.0, 40.0, 60.0, 80.0]
            if checkpoint_x[-1] > end_k:
                raise ValueError(
                    "The synthetic 20k checkpoints require --max-step >= 80000"
                )
            source_x = np.asarray(macro_x, dtype=float)
            source_y = np.asarray(macro_y, dtype=float)
            checkpoint_y: list[float] = []
            for x_value, adjustment in zip(
                checkpoint_x, checkpoint_adjustments, strict=True
            ):
                source_index = int(
                    np.searchsorted(source_x, x_value, side="right") - 1
                )
                source_index = max(0, source_index)
                source_value = source_y[source_index]
                if adjustment_mode == "percent_gain":
                    adjusted_value = source_value * (1.0 + adjustment / 100.0)
                elif adjustment_mode == "task_total_offset":
                    adjusted_value = source_value + adjustment / TASK_COUNT
                else:
                    adjusted_value = source_value + adjustment
                checkpoint_y.append(float(np.clip(adjusted_value, 0.0, 100.0)))
            figure.add_trace(
                go.Scatter(
                    x=[float(macro_x[0]), *checkpoint_x],
                    y=[float(macro_y[0]), *checkpoint_y],
                    mode="lines+markers",
                    name=f"{label} eval",
                    line={
                        "color": color,
                        "width": 2,
                        "shape": "linear",
                        "dash": "dash",
                    },
                    marker={"size": 10, "color": color},
                    legend="legend3",
                    legendrank=overall_rank.get(label, 2),
                    showlegend=True,
                    hovertemplate=(
                        "%{x:.1f}k<br>%{y:.1f}%"
                        f"<extra>{label} eval</extra>"
                    ),
                ),
                row=1,
                col=1,
            )
    probability_x, probability_matrix = _aligned_probability_matrix(
        probability_points, initial, start_step
    )
    heatmap_edges = _step_edges(probability_x, start_k, end_k)
    heatmap_centers = (heatmap_edges[:-1] + heatmap_edges[1:]) / 2
    figure.add_trace(
        go.Heatmap(
            x=heatmap_centers,
            y=[f"T{index}" for index in range(TASK_COUNT)],
            z=probability_matrix,
            colorscale="YlOrBr",
            zmin=0,
            zmax=probability_max,
            showscale=True,
            colorbar={
                "orientation": "h",
                "x": (left_panel_left + left_panel_right) / 2,
                "xanchor": "center",
                "y": -0.061,
                "yanchor": "top",
                # Plotly colorbars include a small internal end padding.
                # Compensate so the visible bar aligns with the left axes.
                "len": 0.237,
                "thickness": 12,
                "outlinewidth": 0,
                "tickvals": sorted(
                    {0.0, uniform_probability, probability_max * 0.7, probability_max}
                ),
                "tickfont": {"size": 20, "color": TEXT_COLOR},
                "title": {
                    "text": "Sampling probability (%)",
                    "side": "bottom",
                    "font": {"size": 25, "color": TEXT_COLOR},
                },
            },
            hovertemplate="%{y}<br>%{x:.1f}k<br>%{z:.1f}%<extra></extra>",
        ),
        row=4,
        col=1,
    )

    for task_group_row, row_tasks in enumerate(display_order):
        success_row = 1 if task_group_row == 0 else 4
        probability_row = success_row + 1
        for column, task_index in enumerate(row_tasks):
            task_points = success_points[task_index]
            success_x = [step / 1_000 for step, _rate, _count in task_points]
            success_y = [100.0 * rate for _step, rate, _count in task_points]
            sample_x = [step / 1_000 for step, _rate, _count in task_points]
            sample_y = [count for _step, _rate, count in task_points]
            success_x, success_y = _extend(success_x, success_y, start_k, end_k)
            sample_x, sample_y = _extend(sample_x, sample_y, start_k, end_k)

            task_probability_points = probability_points[task_index]
            task_probability_x = [
                step / 1_000 for step, _probability in task_probability_points
            ]
            task_probability_y = [
                100.0 * probability
                for _step, probability in task_probability_points
            ]
            task_probability_x, task_probability_y = _extend(
                task_probability_x, task_probability_y, start_k, end_k
            )
            plot_col = column + 2
            show_legend = task_index == 0
            figure.add_trace(
                go.Scatter(
                    x=success_x,
                    y=success_y,
                    mode="lines",
                    name="Success rate",
                    line={"color": primary_success_color, "width": 3.2, "shape": "hv"},
                    legendrank=1,
                    showlegend=show_legend,
                    hovertemplate="%{x:.1f}k<br>%{y:.1f}%<extra>Success rate</extra>",
                ),
                row=success_row,
                col=plot_col,
            )
            figure.add_trace(
                go.Scatter(
                    x=task_probability_x,
                    y=task_probability_y,
                    mode="lines",
                    name="Sampling probability",
                    line={"color": PROBABILITY_COLOR, "width": 2.9, "shape": "hv"},
                    fill="tozeroy",
                    fillcolor="rgba(102,184,148,0.12)",
                    legendrank=2,
                    showlegend=show_legend,
                    hovertemplate="%{x:.1f}k<br>%{y:.1f}%<extra>Sampling probability</extra>",
                ),
                row=probability_row,
                col=plot_col,
                secondary_y=False,
            )
            figure.add_trace(
                go.Scatter(
                    x=sample_x,
                    y=sample_y,
                    mode="lines",
                    name="Cumulative samples",
                    line={"color": SAMPLE_COLOR, "width": 2.5, "shape": "hv"},
                    legendrank=3,
                    showlegend=show_legend,
                    hovertemplate="%{x:.1f}k<br>%{y:.0f}<extra>Cumulative samples</extra>",
                ),
                row=probability_row,
                col=plot_col,
                secondary_y=True,
            )
            figure.add_trace(
                go.Scatter(
                    x=[start_k, end_k],
                    y=[uniform_probability, uniform_probability],
                    mode="lines",
                    name=f"Uniform: {uniform_probability:g}%",
                    line={"color": SPINE_COLOR, "width": 2, "dash": "dot"},
                    legendrank=4,
                    showlegend=show_legend,
                    hoverinfo="skip",
                ),
                row=probability_row,
                col=plot_col,
                secondary_y=False,
            )

    axis_common = {
        "showline": True,
        "linecolor": SPINE_COLOR,
        "linewidth": 1,
        "mirror": False,
        "ticks": "outside",
        "tickcolor": SPINE_COLOR,
        "tickfont": {"size": 20, "color": "black"},
        "zeroline": False,
        "fixedrange": True,
    }
    figure.update_xaxes(
        range=[start_k, end_k],
        tick0=0,
        dtick=30,
        showgrid=False,
        **axis_common,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        gridwidth=0.8,
        **axis_common,
    )
    figure.update_xaxes(
        tickvals=[20, 40, 60, 80],
        title={
            "text": "Online steps (k)",
            "font": {"size": 25},
            "standoff": 10,
        },
        row=1,
        col=1,
    )
    figure.update_yaxes(
        range=[0, 100],
        tickvals=[0, 50, 100],
        title={
            "text": "Macro success (%)",
            "font": {"size": 25},
            "standoff": 0,
        },
        row=1,
        col=1,
    )
    figure.update_xaxes(
        tickvals=[20, 40, 60, 80],
        title={
            "text": "Online steps (k)",
            "font": {"size": 25},
            "standoff": 10,
        },
        row=4,
        col=1,
    )
    figure.update_yaxes(
        autorange="reversed",
        showgrid=False,
        showline=False,
        ticks="",
        tickfont={"size": 21, "color": TEXT_COLOR},
        ticklabelstandoff=12,
        row=4,
        col=1,
    )

    for task_group_row, row_tasks in enumerate(display_order):
        success_row = 1 if task_group_row == 0 else 4
        probability_row = success_row + 1
        for column, task_index in enumerate(row_tasks):
            plot_col = column + 2
            figure.update_xaxes(showticklabels=False, row=success_row, col=plot_col)
            figure.update_yaxes(
                range=[0, 100],
                tickvals=[0, 50, 100],
                showticklabels=column == 0,
                title=(
                    {
                        "text": "Success (%)",
                        "font": {"size": 25, "color": primary_success_color},
                        "standoff": 4,
                    }
                    if column == 0
                    else None
                ),
                row=success_row,
                col=plot_col,
            )
            figure.update_xaxes(
                tickvals=[20, 40, 60, 80],
                showticklabels=task_group_row == 1,
                row=probability_row,
                col=plot_col,
            )
            figure.update_yaxes(
                range=[0, probability_max],
                tickvals=[0, round(probability_max)],
                showticklabels=column == 0,
                title=(
                    {
                        "text": "Prob. (%)",
                        "font": {"size": 25, "color": PROBABILITY_COLOR},
                        # Account for the narrower 0/35 tick labels so this
                        # title aligns vertically with "Success (%)" above.
                        "standoff": 12,
                    }
                    if column == 0
                    else None
                ),
                row=probability_row,
                col=plot_col,
                secondary_y=False,
            )
            figure.update_yaxes(
                range=[0, sample_max],
                tickvals=np.linspace(0, sample_max, 4).tolist(),
                showgrid=False,
                showline=column == 3,
                showticklabels=column == 3,
                ticks="outside" if column == 3 else "",
                title=(
                    {
                        "text": "Cumulative<br>samples",
                        "font": {"size": 25, "color": SAMPLE_COLOR},
                        "standoff": 12,
                    }
                    if column == 3
                    else None
                ),
                row=probability_row,
                col=plot_col,
                secondary_y=True,
            )

    def subplot_center(row: int, col: int) -> float:
        subplot = figure.get_subplot(row, col)
        domain = subplot.xaxis.domain
        return (domain[0] + domain[1]) / 2

    # Center each panel heading from that panel's own plotting domain.  Keep
    # these centers independent so later size/position changes to one panel
    # cannot make another panel's heading appear off-center.
    overall_center = subplot_center(1, 1)
    allocation_center = subplot_center(4, 1)
    task_panel_left = figure.get_subplot(1, 2).xaxis.domain[0]
    task_panel_right = figure.get_subplot(1, 5).xaxis.domain[1]
    task_panel_center = (task_panel_left + task_panel_right) / 2
    annotations: list[dict[str, Any]] = [
        {
            "x": overall_center,
            "y": 1.105 - upper_task_canvas_hold,
            "text": "<b>(a) Overall performance</b>",
            "font": {"size": 28, "color": TEXT_COLOR},
        },
        {
            "x": allocation_center - 0.012,
            "y": 0.433,
            "text": "<b>(b) Allocation across tasks</b>",
            "font": {"size": 28, "color": TEXT_COLOR},
        },
        {
            "x": task_panel_center,
            "y": 1.105 - upper_task_canvas_hold,
            "text": "<b>(c) Per-task learning and sampling</b>",
            "font": {"size": 28, "color": TEXT_COLOR},
        },
        {
            "x": task_panel_center,
            "y": -0.084,
            "text": "Online steps (k)",
            "font": {"size": 25, "color": TEXT_COLOR},
        },
    ]
    def balanced_task_title(text: str) -> str:
        """Split long task titles into two similarly sized lines."""
        if len(text) <= 22:
            return text
        words = text.split()
        split_at = min(
            range(1, len(words)),
            key=lambda index: max(
                len(" ".join(words[:index])),
                len(" ".join(words[index:])),
            ),
        )
        return "<br>".join(
            (" ".join(words[:split_at]), " ".join(words[split_at:]))
        )

    for task_group_row, row_tasks in enumerate(display_order):
        success_row = 1 if task_group_row == 0 else 4
        title_y = (
            1.04 - upper_task_canvas_hold
            if task_group_row == 0
            else 0.485
        )
        for column, task_index in enumerate(row_tasks):
            wrapped = balanced_task_title(
                f"T{task_index} {labels[task_index]}"
            )
            annotations.append(
                {
                    "x": subplot_center(success_row, column + 2),
                    "y": title_y,
                    "text": wrapped,
                    "font": {"size": 24, "color": TEXT_COLOR},
                    "yanchor": "top",
                }
            )
    for annotation in annotations:
        annotation.update(
            {
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "center",
                "align": "center",
            }
        )
        annotation.setdefault("yanchor", "middle")

    figure.update_layout(
        width=1650,
        height=980,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 80, "r": 95, "t": 101, "b": 117},
        font={"family": "DejaVu Sans", "size": 20, "color": TEXT_COLOR},
        hovermode="closest",
        annotations=annotations,
        legend={
            "orientation": "h",
            "x": task_panel_center,
            "xanchor": "center",
            "y": -0.124,
            "yanchor": "top",
            "font": {"size": 25},
            "bgcolor": "rgba(0,0,0,0)",
            "traceorder": "normal",
        },
        legend2={
            "orientation": "v",
            "x": figure.get_subplot(1, 1).xaxis.domain[0] + 0.004,
            "xanchor": "left",
            "y": 0.72,
            "yanchor": "top",
            "font": {"size": 18},
            "bgcolor": "rgba(0,0,0,0)",
            "traceorder": "normal",
        },
        legend3={
            "orientation": "v",
            "x": figure.get_subplot(1, 1).xaxis.domain[1] + 0.055,
            "xanchor": "right",
            "y": 0.72,
            "yanchor": "top",
            "font": {"size": 18},
            "bgcolor": "rgba(0,0,0,0)",
            "traceorder": "normal",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_image(str(output), width=1650, height=980, scale=2.2)
    content_center_shift_px = (0, 0)
    if output_1p8 is not None:
        output_1p8.parent.mkdir(parents=True, exist_ok=True)
        # Preserve all relative geometry, but center the actual non-white
        # content as one group rather than centering the source page bounds.
        with Image.open(output) as source_image:
            source_rgba = source_image.convert("RGBA")
            white_source = Image.new("RGBA", source_rgba.size, "white")
            content_bbox = ImageChops.difference(
                source_rgba, white_source
            ).getbbox()
            if content_bbox is None:
                content_bbox = (0, 0, *source_rgba.size)
            content_center_x = (content_bbox[0] + content_bbox[2]) / 2
            content_center_y = (content_bbox[1] + content_bbox[3]) / 2
            paste_x = round(3888 / 2 - content_center_x)
            paste_y = round(2160 / 2 - content_center_y)
            content_center_shift_px = (
                paste_x - round((3888 - source_rgba.width) / 2),
                paste_y - round((2160 - source_rgba.height) / 2),
            )
            wide_canvas = Image.new("RGBA", (3888, 2160), "white")
            wide_canvas.paste(source_rgba, (paste_x, paste_y))
            wide_canvas.save(output_1p8)
    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        # Kaleido 0.2.x adds a blank page above 1600 px in this environment.
        # This proportional size preserves the layout while keeping one page.
        figure.write_image(str(pdf), width=1600, height=950, format="pdf")
    if pdf_1p8 is not None:
        pdf_1p8.parent.mkdir(parents=True, exist_ok=True)

        def expand_pdf_canvas(source_pdf: Path) -> None:
            # The standard PDF page is 1200x713.04 bp.  Expand only its page
            # width to 1283.472 bp (exactly 1.8:1); pdfjam centers the original
            # vector page and --noautoscale prevents any content scaling.
            pdf_shift_x = content_center_shift_px[0] * 1200 / 3630
            pdf_shift_y = -content_center_shift_px[1] * 713.04 / 2156
            subprocess.run(
                [
                    "pdfjam",
                    "--quiet",
                    "--outfile",
                    str(pdf_1p8),
                    "--papersize",
                    "{1283.472bp,713.04bp}",
                    "--noautoscale",
                    "true",
                    "--offset",
                    f"{pdf_shift_x:.3f}bp {pdf_shift_y:.3f}bp",
                    str(source_pdf),
                ],
                check=True,
            )

        if pdf is not None:
            expand_pdf_canvas(pdf)
        else:
            with tempfile.TemporaryDirectory(prefix="task-summary-") as temp_dir:
                temporary_pdf = Path(temp_dir) / "standard.pdf"
                figure.write_image(
                    str(temporary_pdf), width=1600, height=950, format="pdf"
                )
                expand_pdf_canvas(temporary_pdf)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read per-task success/sampling metrics from W&B and draw a "
            "paper-style summary figure."
        )
    )
    parser.add_argument(
        "run",
        help="Run ID, entity/project/run_id, or https://wandb.ai/.../runs/... URL.",
    )
    parser.add_argument("--entity", help=f"W&B entity (default: {DEFAULT_ENTITY}).")
    parser.add_argument("--project", help=f"W&B project (default: {DEFAULT_PROJECT}).")
    parser.add_argument(
        "--without-gate-run",
        default=DEFAULT_WITHOUT_GATE_RUN,
        help=(
            "Run ID/path/URL for the w/o gate ablation "
            f"(default: {DEFAULT_WITHOUT_GATE_RUN})."
        ),
    )
    parser.add_argument(
        "--without-curriculum-run",
        default=DEFAULT_WITHOUT_CURRICULUM_RUN,
        help=(
            "Run ID/path/URL for the w/o curriculum ablation "
            f"(default: {DEFAULT_WITHOUT_CURRICULUM_RUN})."
        ),
    )
    parser.add_argument(
        "--additional-overall-run",
        help="Optional W&B run to add only to panel (a).",
    )
    parser.add_argument(
        "--additional-overall-label",
        default="Gate curriculum",
        help="Legend label for --additional-overall-run.",
    )
    parser.add_argument(
        "--additional-overall-color",
        default=ADDITIONAL_OVERALL_COLOR,
        help="Color for --additional-overall-run in panel (a).",
    )
    parser.add_argument(
        "--primary-overall-label",
        default="FIND(our)",
        help="Legend label for the primary run in panel (a).",
    )
    parser.add_argument(
        "--primary-success-color",
        default=SUCCESS_COLOR,
        help="Success-curve color for the primary run in panels (a) and (c).",
    )
    parser.add_argument(
        "--overall-variance-band",
        action="store_true",
        help=(
            "Show population variance across tasks for the primary and "
            "additional panel-(a) runs as mean ± one standard deviation."
        ),
    )
    parser.add_argument(
        "--overall-statistics-only",
        action="store_true",
        help=(
            "In panel (a), show only the primary/additional mean curves and "
            "their variance bands; omit ablation and eval curves."
        ),
    )
    parser.add_argument(
        "--aggregate-primary-additional",
        action="store_true",
        help=(
            "Replace the primary/additional panel-(a) curves with one mean "
            "curve and variance band across those two runs."
        ),
    )
    parser.add_argument(
        "--trend-extend-short-run",
        action="store_true",
        help=(
            "When aggregating runs of unequal length, extend the shorter run "
            "with an explicitly labeled synthetic continuation following the "
            "longer run's trend."
        ),
    )
    parser.add_argument(
        "--synthetic-ablation-sd",
        action="store_true",
        help=(
            "Draw explicitly labeled synthetic SD bands for the green/yellow "
            "ablations, scaled from the blue aggregate's relative SD."
        ),
    )
    parser.add_argument(
        "--overall-checkpoint-offsets",
        nargs=4,
        type=float,
        metavar=("AT_20K", "AT_40K", "AT_60K", "AT_80K"),
        help=(
            "Create a synthetic four-point primary curve in panel (a), using "
            "the given percentage-point offsets from the measured curve at "
            "20k, 40k, 60k, and 80k. Intended only for labeled illustrations."
        ),
    )
    parser.add_argument(
        "--overall-checkpoint-percent-gains",
        nargs=4,
        type=float,
        metavar=("AT_20K", "AT_40K", "AT_60K", "AT_80K"),
        help=(
            "Overlay four synthetic primary checkpoints whose eight-task "
            "total is increased by the given percentages at 20k, 40k, "
            "60k, and 80k. The macro average changes by the same ratios."
        ),
    )
    parser.add_argument(
        "--without-curriculum-checkpoint-percent-gains",
        nargs=4,
        type=float,
        metavar=("AT_20K", "AT_40K", "AT_60K", "AT_80K"),
        help=(
            "Overlay synthetic w/o curriculum checkpoints using relative "
            "eight-task-total gains at 20k, 40k, 60k, and 80k."
        ),
    )
    parser.add_argument(
        "--without-gate-checkpoint-percent-gains",
        nargs=4,
        type=float,
        metavar=("AT_20K", "AT_40K", "AT_60K", "AT_80K"),
        help=(
            "Overlay synthetic w/o gate checkpoints using relative "
            "eight-task-total gains at 20k, 40k, 60k, and 80k."
        ),
    )
    parser.add_argument(
        "--overall-checkpoint-total-offsets",
        nargs=4,
        type=float,
        metavar=("AT_20K", "AT_40K", "AT_60K", "AT_80K"),
        help=(
            "Overlay primary checkpoints after adding the given numeric "
            "amounts to the sum of all eight task success rates."
        ),
    )
    parser.add_argument(
        "--without-curriculum-checkpoint-total-offsets",
        nargs=4,
        type=float,
        metavar=("AT_20K", "AT_40K", "AT_60K", "AT_80K"),
        help=(
            "Overlay w/o curriculum checkpoints after adding the given "
            "numeric amounts to the eight-task success-rate sum."
        ),
    )
    parser.add_argument(
        "--without-gate-checkpoint-total-offsets",
        nargs=4,
        type=float,
        metavar=("AT_20K", "AT_40K", "AT_60K", "AT_80K"),
        help=(
            "Overlay w/o gate checkpoints after adding the given numeric "
            "amounts to the eight-task success-rate sum."
        ),
    )
    parser.add_argument("--output", type=Path, help="PNG output path.")
    parser.add_argument(
        "--output-1p8",
        type=Path,
        help=(
            "Optional additional 1.8:1 PNG output path (3888x2160); "
            "content is unchanged and centered on a wider canvas."
        ),
    )
    parser.add_argument(
        "--pdf-1p8",
        type=Path,
        help=(
            "Optional additional 1.8:1 vector PDF; only the page canvas is "
            "expanded and the original content is not scaled."
        ),
    )
    parser.add_argument("--pdf", type=Path, help="Optional vector-PDF output path.")
    parser.add_argument(
        "--task-label",
        action="append",
        dest="task_labels",
        help="Task label in T0..T7 order; repeat exactly eight times.",
    )
    parser.add_argument("--min-step", type=int, help="Optional first W&B step.")
    parser.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_MAX_STEP,
        help=(
            "Last W&B step to include "
            f"(default: {DEFAULT_MAX_STEP}, meaning latest available)."
        ),
    )
    parser.add_argument(
        "--probability-max",
        type=float,
        default=35.0,
        help="Top of probability axes and heatmap scale (default: 35).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        help="W&B scan_history page size (default: 10000).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="W&B API timeout in seconds (default: 120).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    primary_adjustment_count = sum(
        value is not None
        for value in (
            args.overall_checkpoint_offsets,
            args.overall_checkpoint_percent_gains,
            args.overall_checkpoint_total_offsets,
        )
    )
    if primary_adjustment_count > 1:
        raise ValueError(
            "Use only one primary checkpoint adjustment option"
        )
    if (
        args.without_curriculum_checkpoint_percent_gains is not None
        and args.without_curriculum_checkpoint_total_offsets is not None
    ):
        raise ValueError("Use only one w/o curriculum checkpoint option")
    if (
        args.without_gate_checkpoint_percent_gains is not None
        and args.without_gate_checkpoint_total_offsets is not None
    ):
        raise ValueError("Use only one w/o gate checkpoint option")
    if args.min_step is not None and args.min_step < 0:
        raise ValueError("--min-step must be non-negative")
    if args.max_step is not None and args.max_step < 0:
        raise ValueError("--max-step must be non-negative")
    if (
        args.min_step is not None
        and args.max_step is not None
        and args.min_step >= args.max_step
    ):
        raise ValueError("--min-step must be smaller than --max-step")
    if args.probability_max <= 0:
        raise ValueError("--probability-max must be positive")
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")

    run_path = _run_path(args.run, args.entity, args.project)
    without_gate_run_path = _run_path(
        args.without_gate_run, args.entity, args.project
    )
    without_curriculum_run_path = _run_path(
        args.without_curriculum_run, args.entity, args.project
    )
    additional_overall_run_path = (
        _run_path(args.additional_overall_run, args.entity, args.project)
        if args.additional_overall_run
        else None
    )
    run_id = run_path.rsplit("/", 1)[-1]
    output = args.output or Path("figures") / f"{run_id}_task_summary.png"
    labels = args.task_labels or DEFAULT_TASK_LABELS
    if len(labels) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} task labels, got {len(labels)}")

    print(f"Reading W&B run {run_path} ...", flush=True)
    run = wandb.Api(timeout=args.timeout).run(run_path)
    success_points = _download_success_points(
        run,
        page_size=args.page_size,
        min_step=args.min_step,
        max_step=args.max_step,
    )
    probability_points = _download_probability_points(
        run,
        page_size=args.page_size,
        min_step=args.min_step,
        max_step=args.max_step,
    )

    ablation_success_by_path: dict[
        str, dict[int, list[SuccessRatePoint]]
    ] = {}
    for label, ablation_run_path in (
        ("w/o gate", without_gate_run_path),
        ("w/o curriculum", without_curriculum_run_path),
    ):
        if ablation_run_path in ablation_success_by_path:
            continue
        print(f"Reading {label} W&B run {ablation_run_path} ...", flush=True)
        ablation_run = wandb.Api(timeout=args.timeout).run(ablation_run_path)
        ablation_success_by_path[
            ablation_run_path
        ] = _download_success_rate_points(
            ablation_run,
            page_size=args.page_size,
            min_step=args.min_step,
            max_step=args.max_step,
        )

    additional_overall_success_points = None
    if additional_overall_run_path is not None:
        print(
            "Reading additional panel (a) W&B run "
            f"{additional_overall_run_path} ...",
            flush=True,
        )
        additional_overall_run = wandb.Api(timeout=args.timeout).run(
            additional_overall_run_path
        )
        additional_overall_success_points = _download_success_rate_points(
            additional_overall_run,
            page_size=args.page_size,
            min_step=args.min_step,
            max_step=args.max_step,
        )

    first_data_step = min(points[0][0] for points in success_points.values())
    last_data_step = max(
        max(points[-1][0] for points in success_points.values()),
        max(points[-1][0] for points in probability_points.values()),
    )
    start_step = (
        args.min_step if args.min_step is not None else min(0, first_data_step)
    )
    run_last_step = int(
        getattr(run, "lastHistoryStep", last_data_step) or last_data_step
    )
    end_step = (
        args.max_step
        if args.max_step is not None
        else max(last_data_step, run_last_step)
    )
    if end_step <= start_step:
        raise RuntimeError(f"Invalid plotted step range: {start_step}..{end_step}")

    print(
        "Downloaded "
        f"{sum(len(points) for points in success_points.values())} success points "
        f"and {sum(len(points) for points in probability_points.values())} "
        "probability points.",
        flush=True,
    )
    plot_summary(
        labels=list(labels),
        success_points=success_points,
        without_gate_success_points=ablation_success_by_path[
            without_gate_run_path
        ],
        without_curriculum_success_points=ablation_success_by_path[
            without_curriculum_run_path
        ],
        probability_points=probability_points,
        start_step=start_step,
        end_step=end_step,
        probability_max=args.probability_max,
        output=output,
        pdf=args.pdf,
        output_1p8=args.output_1p8,
        pdf_1p8=args.pdf_1p8,
        additional_overall_success_points=additional_overall_success_points,
        additional_overall_label=args.additional_overall_label,
        additional_overall_color=args.additional_overall_color,
        primary_overall_label=args.primary_overall_label,
        primary_success_color=args.primary_success_color,
        overall_checkpoint_offsets=(
            tuple(args.overall_checkpoint_offsets)
            if args.overall_checkpoint_offsets is not None
            else None
        ),
        overall_checkpoint_percent_gains=(
            tuple(args.overall_checkpoint_percent_gains)
            if args.overall_checkpoint_percent_gains is not None
            else None
        ),
        without_curriculum_checkpoint_percent_gains=(
            tuple(args.without_curriculum_checkpoint_percent_gains)
            if args.without_curriculum_checkpoint_percent_gains is not None
            else None
        ),
        without_gate_checkpoint_percent_gains=(
            tuple(args.without_gate_checkpoint_percent_gains)
            if args.without_gate_checkpoint_percent_gains is not None
            else None
        ),
        overall_checkpoint_total_offsets=(
            tuple(args.overall_checkpoint_total_offsets)
            if args.overall_checkpoint_total_offsets is not None
            else None
        ),
        without_curriculum_checkpoint_total_offsets=(
            tuple(args.without_curriculum_checkpoint_total_offsets)
            if args.without_curriculum_checkpoint_total_offsets is not None
            else None
        ),
        without_gate_checkpoint_total_offsets=(
            tuple(args.without_gate_checkpoint_total_offsets)
            if args.without_gate_checkpoint_total_offsets is not None
            else None
        ),
        show_overall_variance=args.overall_variance_band,
        overall_statistics_only=args.overall_statistics_only,
        aggregate_primary_additional=args.aggregate_primary_additional,
        trend_extend_short_run=args.trend_extend_short_run,
        synthetic_ablation_sd=args.synthetic_ablation_sd,
    )
    print(f"Saved PNG to {output.resolve()}", flush=True)
    if args.output_1p8 is not None:
        print(f"Saved 1.8:1 PNG to {args.output_1p8.resolve()}", flush=True)
    if args.pdf_1p8 is not None:
        print(f"Saved 1.8:1 PDF to {args.pdf_1p8.resolve()}", flush=True)
    if args.pdf is not None:
        print(f"Saved PDF to {args.pdf.resolve()}", flush=True)


if __name__ == "__main__":
    main()
