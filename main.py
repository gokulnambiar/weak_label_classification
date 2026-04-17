from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.classifier import fit_vectorizer, predict_labels, train_linear_svm
from src.data_loader import LABEL_NAMES, ensure_dataset_downloaded, load_ag_news_splits, select_labeled_subset
from src.evaluation import (
    build_confusion_frame,
    build_label_distribution,
    compute_classification_metrics,
    compute_per_class_metrics,
    select_label_examples,
    summarize_assigned_labels,
)
from src.label_refinement import estimate_rule_precisions, refine_weak_labels
from src.visualization import (
    plot_confusion_matrix,
    plot_label_distribution,
    plot_label_quality_summary,
    plot_scenario_comparison,
)
from src.weak_labeling import apply_labeling_rules, build_default_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the weak-label AG News classification benchmark.")
    parser.add_argument("--labeled-fraction", type=float, default=0.02, help="Fraction of gold training labels to expose.")
    parser.add_argument("--max-features", type=int, default=30000, help="Maximum number of TF-IDF features.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used for subset selection and model training.")
    parser.add_argument("--force-download", action="store_true", help="Redownload the AG News raw files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_dataset_downloaded(data_dir=data_dir, force=args.force_download)
    train_frame, test_frame = load_ag_news_splits(data_dir=data_dir)
    labeled_subset, unlabeled_pool = select_labeled_subset(
        train_frame=train_frame,
        labeled_fraction=args.labeled_fraction,
        seed=args.seed,
    )

    rules = build_default_rules()
    subset_weak = apply_labeling_rules(labeled_subset["text"].tolist(), rules, num_classes=len(LABEL_NAMES))
    pool_weak = apply_labeling_rules(unlabeled_pool["text"].tolist(), rules, num_classes=len(LABEL_NAMES))

    rule_diagnostics = estimate_rule_precisions(
        rule_matrix=subset_weak.rule_matrix,
        rules=rules,
        gold_labels=labeled_subset["label_id"].to_numpy(),
    )
    rule_weights = dict(zip(rule_diagnostics["rule_name"], rule_diagnostics["precision"]))

    subset_refined = refine_weak_labels(
        rule_matrix=subset_weak.rule_matrix,
        rules=rules,
        rule_weights=rule_weights,
        num_classes=len(LABEL_NAMES),
    )
    pool_refined = refine_weak_labels(
        rule_matrix=pool_weak.rule_matrix,
        rules=rules,
        rule_weights=rule_weights,
        num_classes=len(LABEL_NAMES),
    )

    weak_quality = summarize_assigned_labels(
        gold_labels=labeled_subset["label_id"].to_numpy(),
        assigned_labels=subset_weak.weak_labels,
        label_names=LABEL_NAMES,
        scenario="Weak rules",
    )
    refined_quality = summarize_assigned_labels(
        gold_labels=labeled_subset["label_id"].to_numpy(),
        assigned_labels=subset_refined.refined_labels,
        label_names=LABEL_NAMES,
        scenario="Refined rules",
    )
    weak_quality["conflict_rate"] = float(subset_weak.conflict.mean())
    refined_quality["conflict_rate"] = float(((subset_refined.class_scores > 0).sum(axis=1) > 1).mean())
    label_quality_frame = pd.DataFrame([weak_quality, refined_quality])

    distribution_frame = pd.concat(
        [
            build_label_distribution(train_frame["label_id"].to_numpy(), LABEL_NAMES, "Gold train"),
            build_label_distribution(pool_weak.weak_labels, LABEL_NAMES, "Weak labels"),
            build_label_distribution(pool_refined.refined_labels, LABEL_NAMES, "Refined labels"),
        ],
        ignore_index=True,
    )

    label_examples = select_label_examples(
        labeled_subset=labeled_subset,
        weak_labels=subset_weak.weak_labels,
        refined_labels=subset_refined.refined_labels,
        weak_confidence=subset_weak.confidence,
        refined_confidence=subset_refined.confidence,
        label_names=LABEL_NAMES,
    )

    vectorizer = fit_vectorizer(train_frame["text"].tolist(), max_features=args.max_features)
    train_features = vectorizer.transform(train_frame["text"])
    test_features = vectorizer.transform(test_frame["text"])

    full_train_labels = train_frame["label_id"].to_numpy()
    weak_train_indices = np.concatenate(
        [
            labeled_subset.index.to_numpy(),
            unlabeled_pool.index.to_numpy()[pool_weak.weak_labels != -1],
        ]
    )
    weak_train_labels = np.concatenate(
        [
            labeled_subset["label_id"].to_numpy(),
            pool_weak.weak_labels[pool_weak.weak_labels != -1],
        ]
    )
    refined_train_indices = np.concatenate(
        [
            labeled_subset.index.to_numpy(),
            unlabeled_pool.index.to_numpy()[pool_refined.refined_labels != -1],
        ]
    )
    refined_train_labels = np.concatenate(
        [
            labeled_subset["label_id"].to_numpy(),
            pool_refined.refined_labels[pool_refined.refined_labels != -1],
        ]
    )

    full_model = train_linear_svm(train_features, full_train_labels, seed=args.seed)
    weak_model = train_linear_svm(train_features[weak_train_indices], weak_train_labels, seed=args.seed)
    refined_model = train_linear_svm(train_features[refined_train_indices], refined_train_labels, seed=args.seed)

    test_labels = test_frame["label_id"].to_numpy()
    full_predictions = predict_labels(full_model, test_features)
    weak_predictions = predict_labels(weak_model, test_features)
    refined_predictions = predict_labels(refined_model, test_features)

    metrics_frame = pd.DataFrame(
        [
            compute_classification_metrics(test_labels, full_predictions, "Full supervision"),
            compute_classification_metrics(test_labels, weak_predictions, "Weak supervision"),
            compute_classification_metrics(test_labels, refined_predictions, "Refined weak supervision"),
        ]
    )
    per_class_metrics = pd.concat(
        [
            compute_per_class_metrics(test_labels, full_predictions, LABEL_NAMES, "Full supervision"),
            compute_per_class_metrics(test_labels, weak_predictions, LABEL_NAMES, "Weak supervision"),
            compute_per_class_metrics(test_labels, refined_predictions, LABEL_NAMES, "Refined weak supervision"),
        ],
        ignore_index=True,
    )

    weak_confusion = build_confusion_frame(test_labels, weak_predictions, LABEL_NAMES)
    refined_confusion = build_confusion_frame(test_labels, refined_predictions, LABEL_NAMES)

    metrics_frame.to_csv(output_dir / "scenario_metrics.csv", index=False)
    per_class_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    rule_diagnostics.to_csv(output_dir / "rule_diagnostics.csv", index=False)
    label_quality_frame.to_csv(output_dir / "weak_label_quality.csv", index=False)
    distribution_frame.to_csv(output_dir / "weak_label_distribution.csv", index=False)
    label_examples.to_csv(output_dir / "label_examples.csv", index=False)
    weak_confusion.to_csv(output_dir / "weak_confusion_matrix.csv")
    refined_confusion.to_csv(output_dir / "refined_confusion_matrix.csv")

    plot_scenario_comparison(metrics_frame, output_dir / "performance_comparison.png")
    plot_confusion_matrix(weak_confusion, "Weak supervision confusion matrix", output_dir / "weak_confusion_matrix.png")
    plot_confusion_matrix(
        refined_confusion,
        "Refined weak supervision confusion matrix",
        output_dir / "refined_confusion_matrix.png",
    )
    plot_label_quality_summary(label_quality_frame, output_dir / "weak_label_coverage.png")
    plot_label_distribution(distribution_frame, output_dir / "weak_label_balance.png")

    run_metadata = {
        "dataset": "AG News",
        "train_examples": int(len(train_frame)),
        "test_examples": int(len(test_frame)),
        "labeled_fraction": args.labeled_fraction,
        "labeled_subset_examples": int(len(labeled_subset)),
        "weak_training_examples": int(len(weak_train_labels)),
        "refined_training_examples": int(len(refined_train_labels)),
        "vectorizer_max_features": args.max_features,
        "rule_count": len(rules),
        "seed": args.seed,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics_summary.txt").write_text(
        build_summary_report(
            metrics_frame=metrics_frame,
            label_quality_frame=label_quality_frame,
            per_class_metrics=per_class_metrics,
            weak_train_size=len(weak_train_labels),
            refined_train_size=len(refined_train_labels),
            labeled_subset_size=len(labeled_subset),
        ),
        encoding="utf-8",
    )


def build_summary_report(
    metrics_frame: pd.DataFrame,
    label_quality_frame: pd.DataFrame,
    per_class_metrics: pd.DataFrame,
    weak_train_size: int,
    refined_train_size: int,
    labeled_subset_size: int,
) -> str:
    weak_gap = metrics_frame.loc[metrics_frame["scenario"] == "Full supervision", "macro_f1"].iloc[0] - metrics_frame.loc[
        metrics_frame["scenario"] == "Weak supervision", "macro_f1"
    ].iloc[0]
    recovered_gap = metrics_frame.loc[
        metrics_frame["scenario"] == "Refined weak supervision", "macro_f1"
    ].iloc[0] - metrics_frame.loc[metrics_frame["scenario"] == "Weak supervision", "macro_f1"].iloc[0]

    weak_per_class = per_class_metrics[per_class_metrics["scenario"] == "Weak supervision"].sort_values("f1")
    refined_per_class = per_class_metrics[per_class_metrics["scenario"] == "Refined weak supervision"].sort_values("f1")

    lines = [
        "Weak Label Classification Report",
        "",
        f"Partial gold subset size: {labeled_subset_size}",
        f"Weak supervision training size: {weak_train_size}",
        f"Refined weak supervision training size: {refined_train_size}",
        "",
        "Scenario metrics:",
    ]

    for row in metrics_frame.itertuples(index=False):
        lines.append(
            f"- {row.scenario}: accuracy={row.accuracy:.4f}, macro_f1={row.macro_f1:.4f}, "
            f"macro_precision={row.macro_precision:.4f}, macro_recall={row.macro_recall:.4f}"
        )

    lines.extend(
        [
            "",
            f"Macro F1 drop from full supervision to weak supervision: {weak_gap:.4f}",
            f"Macro F1 recovered by refinement: {recovered_gap:.4f}",
            "",
            "Weak label quality on the labeled subset:",
        ]
    )

    for row in label_quality_frame.itertuples(index=False):
        lines.append(
            f"- {row.scenario}: coverage={row.coverage:.4f}, precision={row.estimated_precision:.4f}, "
            f"conflict_rate={row.conflict_rate:.4f}"
        )

    lines.extend(
        [
            "",
            f"Hardest class under weak supervision: {weak_per_class.iloc[0]['label_name']} "
            f"(F1={weak_per_class.iloc[0]['f1']:.4f})",
            f"Hardest class after refinement: {refined_per_class.iloc[0]['label_name']} "
            f"(F1={refined_per_class.iloc[0]['f1']:.4f})",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
