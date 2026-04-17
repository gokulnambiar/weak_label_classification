from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def plot_scenario_comparison(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(metrics_frame))
    bar_width = 0.35

    axis.bar(x_positions - bar_width / 2, metrics_frame["accuracy"], width=bar_width, label="Accuracy", color="#355070")
    axis.bar(
        x_positions + bar_width / 2,
        metrics_frame["macro_f1"],
        width=bar_width,
        label="Macro F1",
        color="#6d597a",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(metrics_frame["scenario"], rotation=10)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("AG News performance under different supervision settings")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(confusion_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion_frame.values, cmap="Blues", vmin=0.0, vmax=1.0)

    axis.set_xticks(np.arange(len(confusion_frame.columns)))
    axis.set_yticks(np.arange(len(confusion_frame.index)))
    axis.set_xticklabels(confusion_frame.columns, rotation=30, ha="right")
    axis.set_yticklabels(confusion_frame.index)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)

    for row_index in range(confusion_frame.shape[0]):
        for column_index in range(confusion_frame.shape[1]):
            value = confusion_frame.iloc[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "#1f2933",
                fontsize=9,
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_label_quality_summary(summary_frame: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(summary_frame))
    bar_width = 0.35

    axis.bar(
        x_positions - bar_width / 2,
        summary_frame["coverage"],
        width=bar_width,
        label="Coverage",
        color="#457b9d",
    )
    axis.bar(
        x_positions + bar_width / 2,
        summary_frame["estimated_precision"],
        width=bar_width,
        label="Precision on labeled subset",
        color="#e76f51",
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(summary_frame["scenario"], rotation=10)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Share")
    axis.set_title("Weak label quality on the partial gold subset")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_label_distribution(distribution_frame: pd.DataFrame, output_path: Path) -> None:
    pivot_frame = distribution_frame.pivot(index="label_name", columns="scenario", values="share").fillna(0.0)
    figure, axis = plt.subplots(figsize=(8, 5))
    x_positions = np.arange(len(pivot_frame.index))
    scenario_names = list(pivot_frame.columns)
    bar_width = 0.22

    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261"]
    for offset, scenario_name in enumerate(scenario_names):
        axis.bar(
            x_positions + (offset - (len(scenario_names) - 1) / 2) * bar_width,
            pivot_frame[scenario_name],
            width=bar_width,
            label=scenario_name,
            color=colors[offset % len(colors)],
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(pivot_frame.index, rotation=15)
    axis.set_ylim(0.0, 0.5)
    axis.set_ylabel("Share of labeled examples")
    axis.set_title("Class balance under gold, weak, and refined labels")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
