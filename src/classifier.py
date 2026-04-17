from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def build_vectorizer(max_features: int = 30000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
    )


def fit_vectorizer(train_texts: list[str] | np.ndarray, max_features: int = 30000) -> TfidfVectorizer:
    vectorizer = build_vectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    return vectorizer


def train_linear_svm(features, labels: np.ndarray, seed: int) -> LinearSVC:
    classifier = LinearSVC(random_state=seed, dual="auto")
    classifier.fit(features, labels)
    return classifier


def predict_labels(classifier: LinearSVC, features) -> np.ndarray:
    return classifier.predict(features)
