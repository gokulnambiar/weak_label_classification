from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LabelRule:
    name: str
    label_id: int
    keywords: tuple[str, ...]
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class WeakLabelArtifacts:
    weak_labels: np.ndarray
    vote_matrix: np.ndarray
    rule_matrix: np.ndarray
    fired_rules: list[list[str]]
    confidence: np.ndarray
    conflict: np.ndarray


def make_rule(name: str, label_id: int, keywords: Iterable[str]) -> LabelRule:
    normalized_keywords = tuple(sorted({keyword.lower() for keyword in keywords}, key=len, reverse=True))
    pattern_text = "|".join(re.escape(keyword) for keyword in normalized_keywords)
    pattern = re.compile(rf"(?<!\w)(?:{pattern_text})(?!\w)")
    return LabelRule(name=name, label_id=label_id, keywords=normalized_keywords, pattern=pattern)


def build_default_rules() -> list[LabelRule]:
    return [
        make_rule(
            "world_government",
            0,
            [
                "president",
                "prime minister",
                "minister",
                "government",
                "parliament",
                "election",
                "diplomatic",
                "summit",
                "embassy",
            ],
        ),
        make_rule(
            "world_conflict",
            0,
            [
                "war",
                "military",
                "troops",
                "rebels",
                "militant",
                "ceasefire",
                "sanctions",
                "border",
                "peace talks",
            ],
        ),
        make_rule(
            "world_international",
            0,
            [
                "united nations",
                "nato",
                "foreign minister",
                "province",
                "capital city",
                "referendum",
                "coalition",
                "envoy",
            ],
        ),
        make_rule(
            "sports_results",
            1,
            [
                "match",
                "season",
                "tournament",
                "championship",
                "playoff",
                "league",
                "cup",
                "final",
                "semifinal",
            ],
        ),
        make_rule(
            "sports_gameplay",
            1,
            [
                "coach",
                "goal",
                "touchdown",
                "scored",
                "defeated",
                "athlete",
                "victory",
                "quarterback",
                "goalkeeper",
            ],
        ),
        make_rule(
            "sports_leagues",
            1,
            [
                "world cup",
                "olympic",
                "nba",
                "nfl",
                "mlb",
                "fifa",
                "tennis",
                "cricket",
                "grand prix",
            ],
        ),
        make_rule(
            "business_markets",
            2,
            [
                "stocks",
                "shares",
                "investor",
                "trading",
                "wall street",
                "nasdaq",
                "dow",
                "bonds",
                "market",
            ],
        ),
        make_rule(
            "business_company",
            2,
            [
                "earnings",
                "revenue",
                "profit",
                "merger",
                "acquisition",
                "ceo",
                "bank",
                "retailer",
                "quarterly",
            ],
        ),
        make_rule(
            "business_macro",
            2,
            [
                "oil prices",
                "inflation",
                "interest rates",
                "federal reserve",
                "exports",
                "imports",
                "currency",
                "dollar",
                "yen",
            ],
        ),
        make_rule(
            "tech_digital",
            3,
            [
                "software",
                "internet",
                "online",
                "digital",
                "cybersecurity",
                "smartphone",
                "cloud",
                "platform",
                "web site",
            ],
        ),
        make_rule(
            "tech_devices",
            3,
            [
                "chip",
                "semiconductor",
                "computer",
                "laptop",
                "server",
                "wireless",
                "telecom",
                "broadband",
                "search engine",
            ],
        ),
        make_rule(
            "tech_science",
            3,
            [
                "research",
                "scientists",
                "study",
                "nasa",
                "space",
                "satellite",
                "biotech",
                "genome",
                "physics",
            ],
        ),
    ]


def apply_labeling_rules(texts: Iterable[str], rules: list[LabelRule], num_classes: int) -> WeakLabelArtifacts:
    text_list = [text.lower() for text in texts]
    sample_count = len(text_list)
    rule_count = len(rules)

    rule_matrix = np.zeros((sample_count, rule_count), dtype=np.int8)
    vote_matrix = np.zeros((sample_count, num_classes), dtype=np.float32)
    fired_rules: list[list[str]] = []

    for row_index, text in enumerate(text_list):
        matched_rules: list[str] = []
        for rule_index, rule in enumerate(rules):
            if rule.pattern.search(text):
                rule_matrix[row_index, rule_index] = 1
                vote_matrix[row_index, rule.label_id] += 1.0
                matched_rules.append(rule.name)
        fired_rules.append(matched_rules)

    total_votes = vote_matrix.sum(axis=1)
    weak_labels = np.full(sample_count, -1, dtype=int)
    confidence = np.zeros(sample_count, dtype=np.float32)

    labeled_mask = total_votes > 0
    weak_labels[labeled_mask] = vote_matrix[labeled_mask].argmax(axis=1)
    confidence[labeled_mask] = vote_matrix[labeled_mask].max(axis=1) / total_votes[labeled_mask]
    conflict = (vote_matrix > 0).sum(axis=1) > 1

    return WeakLabelArtifacts(
        weak_labels=weak_labels,
        vote_matrix=vote_matrix,
        rule_matrix=rule_matrix,
        fired_rules=fired_rules,
        confidence=confidence,
        conflict=conflict,
    )
