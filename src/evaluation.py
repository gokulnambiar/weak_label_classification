from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def label_name_for_id(label_id: int, label_names: Iterable[str]) -> str:
    label_names_list = list(label_names)
    if label_id == -1:
        return "Unlabeled"
    return label_names_list[label_id]


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scenario: str) -> dict[str, float | str]:
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "scenario": scenario,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    scenario: str,
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "scenario": scenario,
            "label_id": np.arange(len(label_names)),
            "label_name": label_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def build_confusion_frame(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)), normalize="true")
    return pd.DataFrame(matrix, index=label_names, columns=label_names)


def summarize_assigned_labels(
    gold_labels: np.ndarray,
    assigned_labels: np.ndarray,
    label_names: list[str],
    scenario: str,
) -> dict[str, float | int | str]:
    covered_mask = assigned_labels != -1
    covered_count = int(covered_mask.sum())
    coverage = covered_count / len(assigned_labels)
    precision = float((assigned_labels[covered_mask] == gold_labels[covered_mask]).mean()) if covered_count else 0.0

    return {
        "scenario": scenario,
        "covered_examples": covered_count,
        "coverage": coverage,
        "estimated_precision": precision,
        "unlabeled_examples": int((assigned_labels == -1).sum()),
        "conflict_rate": float("nan"),
        "label_space": ", ".join(label_names),
    }


def build_label_distribution(
    labels: np.ndarray,
    label_names: list[str],
    scenario: str,
    include_unlabeled: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    if include_unlabeled:
        target_ids = [-1, *range(len(label_names))]
    else:
        target_ids = list(range(len(label_names)))

    valid_mask = np.ones(len(labels), dtype=bool) if include_unlabeled else labels != -1
    total = int(valid_mask.sum())

    for label_id in target_ids:
        count = int((labels == label_id).sum())
        if not include_unlabeled and label_id == -1:
            continue
        share = (count / total) if total else 0.0
        rows.append(
            {
                "scenario": scenario,
                "label_id": label_id,
                "label_name": label_name_for_id(label_id, label_names),
                "count": count,
                "share": share,
            }
        )

    return pd.DataFrame(rows)


def select_label_examples(
    labeled_subset: pd.DataFrame,
    weak_labels: np.ndarray,
    refined_labels: np.ndarray,
    weak_confidence: np.ndarray,
    refined_confidence: np.ndarray,
    label_names: list[str],
    sample_size: int = 18,
) -> pd.DataFrame:
    examples = labeled_subset.copy()
    examples["true_label"] = examples["label_id"].map(lambda label_id: label_name_for_id(label_id, label_names))
    examples["weak_label"] = [label_name_for_id(label_id, label_names) for label_id in weak_labels]
    examples["refined_label"] = [label_name_for_id(label_id, label_names) for label_id in refined_labels]
    examples["weak_confidence"] = weak_confidence
    examples["refined_confidence"] = refined_confidence

    recovered_mask = (weak_labels != examples["label_id"].to_numpy()) & (
        refined_labels == examples["label_id"].to_numpy()
    )
    persistent_error_mask = refined_labels != examples["label_id"].to_numpy()
    initially_unlabeled_mask = weak_labels == -1

    examples["priority"] = np.select(
        [recovered_mask, persistent_error_mask, initially_unlabeled_mask],
        [0, 1, 2],
        default=3,
    )

    selected_columns = [
        "priority",
        "label_name",
        "true_label",
        "weak_label",
        "refined_label",
        "weak_confidence",
        "refined_confidence",
        "title",
        "description",
        "text",
    ]
    return examples[selected_columns].sort_values(["priority", "label_name", "title"]).head(sample_size).reset_index(drop=True)
