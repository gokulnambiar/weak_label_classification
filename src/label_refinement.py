from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.weak_labeling import LabelRule


@dataclass(frozen=True)
class RefinedLabelArtifacts:
    refined_labels: np.ndarray
    class_scores: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray


def estimate_rule_precisions(
    rule_matrix: np.ndarray,
    rules: list[LabelRule],
    gold_labels: np.ndarray,
    alpha: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    sample_count = len(gold_labels)

    for rule_index, rule in enumerate(rules):
        fired_mask = rule_matrix[:, rule_index].astype(bool)
        fired_count = int(fired_mask.sum())
        correct_count = int((gold_labels[fired_mask] == rule.label_id).sum()) if fired_count else 0
        precision = (correct_count + alpha) / (fired_count + 2 * alpha)
        rows.append(
            {
                "rule_name": rule.name,
                "label_id": rule.label_id,
                "fired_count_on_subset": fired_count,
                "coverage_on_subset": fired_count / sample_count,
                "correct_count_on_subset": correct_count,
                "precision": precision,
            }
        )

    return pd.DataFrame(rows).sort_values(["label_id", "precision"], ascending=[True, False]).reset_index(drop=True)


def refine_weak_labels(
    rule_matrix: np.ndarray,
    rules: list[LabelRule],
    rule_weights: dict[str, float],
    num_classes: int,
    minimum_confidence: float = 0.62,
    minimum_margin: float = 0.15,
) -> RefinedLabelArtifacts:
    sample_count = rule_matrix.shape[0]
    class_scores = np.zeros((sample_count, num_classes), dtype=np.float32)

    for rule_index, rule in enumerate(rules):
        weight = float(rule_weights.get(rule.name, 0.5))
        class_scores[:, rule.label_id] += rule_matrix[:, rule_index] * weight

    refined_labels = np.full(sample_count, -1, dtype=int)
    confidence = np.zeros(sample_count, dtype=np.float32)
    margin = np.zeros(sample_count, dtype=np.float32)

    for row_index in range(sample_count):
        score_row = class_scores[row_index]
        score_total = float(score_row.sum())
        if score_total == 0.0:
            continue

        normalized_scores = score_row / score_total
        ordered_indices = np.argsort(normalized_scores)
        top_index = int(ordered_indices[-1])
        second_index = int(ordered_indices[-2]) if num_classes > 1 else top_index

        confidence[row_index] = float(normalized_scores[top_index])
        margin[row_index] = float(normalized_scores[top_index] - normalized_scores[second_index])

        if confidence[row_index] >= minimum_confidence and margin[row_index] >= minimum_margin:
            refined_labels[row_index] = top_index

    return RefinedLabelArtifacts(
        refined_labels=refined_labels,
        class_scores=class_scores,
        confidence=confidence,
        margin=margin,
    )
