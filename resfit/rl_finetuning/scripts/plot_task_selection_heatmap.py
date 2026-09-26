#!/usr/bin/env python3
"""Plot task-selection probabilities from one or more local W&B sessions."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


PROBABILITY_KEY = re.compile(r"^tasks/task_(\d+)_probability$")


def _history_row(record: wandb_internal_pb2.Record) -> dict[str, object]:
    row: dict[str, object] = {}
    for item in record.history.item:
        key = "/".join(item.nested_key) if item.nested_key else item.key
        row[key] = json.loads(item.value_json)
    return row


def _read_probability_rows(wandb_files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
            if any(PROBABILITY_KEY.fullmatch(key) for key in row):
                rows.append(row)

    # Resumed runs can produce several local files. Keep the newest task record
    # if the same global step appears in more than one session.
    rows.sort(key=lambda row: (int(row["_step"]), float(row.get("_timestamp", 0))))
    by_step = {int(row["_step"]): row for row in rows}
    return [by_step[step] for step in sorted(by_step)]


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


def _step_edges(steps: np.ndarray) -> np.ndarray:
    if len(steps) == 1:
        return np.array([steps[0], steps[0] + 1], dtype=float)
    final_width = float(np.median(np.diff(steps)))
    return np.concatenate((steps.astype(float), [steps[-1] + final_width]))


def _format_step(value: float, _position: int) -> str:
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge local W&B sessions for a run ID and plot "
            "tasks/task_<n>_probability over global step."
        )
    )
    parser.add_argument("--wandb-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path for the merged probability history as CSV.",
    )
    parser.add_argument(
        "--task-label",
        action="append",
        dest="task_labels",
        help="Task label in task-index order; repeat once per task.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color-scale maximum (default: maximum observed probability).",
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

    rows = _read_probability_rows(wandb_files)
    if not rows:
        raise RuntimeError(
            f"Run {args.run_id!r} has no tasks/task_<n>_probability history"
        )

    task_indices = sorted(
        {
            int(match.group(1))
            for row in rows
            for key in row
            if (match := PROBABILITY_KEY.fullmatch(key))
        }
    )
    expected_indices = list(range(len(task_indices)))
    if task_indices != expected_indices:
        raise RuntimeError(
            f"Expected contiguous task indices {expected_indices}, got {task_indices}"
        )

    task_labels = args.task_labels or _infer_task_labels(
        session_dirs, len(task_indices)
    )
    if len(task_labels) != len(task_indices):
        raise ValueError(
            f"Got {len(task_labels)} labels for {len(task_indices)} tasks"
        )

    steps = np.asarray([int(row["_step"]) for row in rows])
    probabilities = np.asarray(
        [
            [float(row[f"tasks/task_{index}_probability"]) for row in rows]
            for index in task_indices
        ]
    )
    probability_sums = probabilities.sum(axis=0)
    if not np.allclose(probability_sums, 1.0, atol=1e-6):
        raise RuntimeError(
            "Task probabilities do not sum to one at every logged step; "
            f"observed range is {probability_sums.min():.6f}--"
            f"{probability_sums.max():.6f}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["global_step", "timestamp"] + [
                f"task_{index}_probability" for index in task_indices
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "global_step": int(row["_step"]),
                        "timestamp": float(row.get("_timestamp", "nan")),
                        **{
                            f"task_{index}_probability": float(
                                row[f"tasks/task_{index}_probability"]
                            )
                            for index in task_indices
                        },
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure_height = max(4.5, 0.62 * len(task_indices))
    figure, axis = plt.subplots(figsize=(16, figure_height), constrained_layout=True)
    mesh = axis.pcolormesh(
        _step_edges(steps),
        np.arange(len(task_indices) + 1),
        probabilities,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=args.vmax,
        rasterized=True,
    )
    axis.set_yticks(np.arange(len(task_indices)) + 0.5, task_labels)
    axis.invert_yaxis()
    axis.set_xlabel("Global training step")
    axis.set_ylabel("Task")
    axis.xaxis.set_major_formatter(FuncFormatter(_format_step))
    axis.set_title(
        f"Task-selection probability over training — W&B run {args.run_id}"
    )
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.015)
    colorbar.set_label("Selection probability")
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(
        f"Read {len(rows)} probability updates from {len(wandb_files)} "
        f"session files (steps {steps[0]}--{steps[-1]})."
    )
    print(f"Saved heatmap to {args.output}")
    if args.csv is not None:
        print(f"Saved merged history to {args.csv}")


if __name__ == "__main__":
    main()
