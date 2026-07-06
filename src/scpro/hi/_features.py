from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from scpro._utils import as_dense

FeatureSelector = Literal["keras_permutation", "decision_tree"]


def _top_features_from_scores(names: np.ndarray, scores: np.ndarray, n_features: int) -> dict[str, float]:
    scores = np.nan_to_num(scores.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    scores[scores < 0] = 0
    total = scores.sum()
    if total > 0:
        scores = scores / total
    order = np.argsort(scores)[::-1][:n_features]
    return {str(names[i]): float(scores[i]) for i in order}


def _keras_permutation_importance(x: np.ndarray, y: np.ndarray, *, seed: int, epochs: int) -> np.ndarray:
    try:
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.models import Sequential
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SCPRO-HI feature_selector='keras_permutation' requires TensorFlow. "
            "Install scpro[hi] or use feature_selector='decision_tree' for a lightweight fallback."
        ) from exc

    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(x.shape[1],)),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x, y, epochs=epochs, batch_size=32, verbose=False)

    pred = model.predict(x, verbose=False).ravel() >= 0.5
    baseline = accuracy_score(y, pred)
    rng = np.random.default_rng(seed)
    importances = np.zeros(x.shape[1], dtype=float)
    # Manual permutation importance avoids relying on Keras behaving as a sklearn estimator.
    for j in range(x.shape[1]):
        x_perm = x.copy()
        rng.shuffle(x_perm[:, j])
        pred_perm = model.predict(x_perm, verbose=False).ravel() >= 0.5
        importances[j] = baseline - accuracy_score(y, pred_perm)
    return importances


def _decision_tree_importance(x: np.ndarray, y: np.ndarray, *, seed: int) -> np.ndarray:
    clf = DecisionTreeClassifier(random_state=seed, max_depth=3)
    clf.fit(x, y)
    return np.asarray(clf.feature_importances_, dtype=float)


def select_distinctive_features(
    adata,
    labels,
    n_features: int,
    *,
    selector: FeatureSelector = "keras_permutation",
    random_state: int = 42,
    epochs: int = 10,
) -> dict[str, float]:
    """Select distinctive proteins/features for a binary cluster label.

    This is the package equivalent of the paper-code `get_HVGs()` routine. The
    default preserves the neural-network + permutation-importance idea but
    computes permutation importance manually to avoid sklearn/Keras estimator
    incompatibilities.
    """
    x = as_dense(adata.X)
    y = np.asarray(labels).astype(int)
    if x.shape[0] != y.shape[0]:
        raise ValueError("labels must have one value per cell.")
    n_features = min(int(n_features), x.shape[1])
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if np.unique(y).size < 2:
        # No classification signal: return features with zero scores in original order.
        return {str(name): 0.0 for name in np.asarray(adata.var_names)[:n_features]}

    if selector == "keras_permutation":
        scores = _keras_permutation_importance(x, y, seed=random_state, epochs=epochs)
    elif selector == "decision_tree":
        scores = _decision_tree_importance(x, y, seed=random_state)
    else:
        raise ValueError(f"Unknown feature selector: {selector}")
    return _top_features_from_scores(np.asarray(adata.var_names), scores, n_features)


def cluster_similarity(features_1: dict[str, float], features_2: dict[str, float]) -> float:
    common = [k for k in features_1 if k in features_2]
    if not common:
        return 0.0
    v1 = np.asarray([features_1[k] for k in common], dtype=float)
    v2 = np.asarray([features_2[k] for k in common], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)
