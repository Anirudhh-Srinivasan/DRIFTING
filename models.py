"""Underlying task classifiers.

These are the models that actually classify traffic (benign vs attack family) --
distinct from the drift-*handling* logic. The prompt specifies lightweight,
standard, cheap-to-retrain models: Logistic Regression and Random Forest. Both
are batch-trained on a window of recent samples; "streaming" is simulated by
processing the stream in order and retraining windows on demand.

A fresh estimator is built per (re)train so training cost is a clean, comparable
wall-clock measurement across pipelines.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import ModelConfig


def make_model(cfg: ModelConfig, seed: int):
    """Return an unfitted sklearn estimator per the config."""
    if cfg.kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, random_state=seed),
        )
    if cfg.kind == "random_forest":
        # n_jobs=1 on purpose: the prequential loop calls predict() once per
        # sample (tens of thousands of times), and joblib's per-call thread
        # dispatch under n_jobs=-1 dwarfs the actual tree traversal for a single
        # row -- single-threaded is dramatically faster here. Fits are few and on
        # small windows, so we don't miss the parallelism.
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model kind: {cfg.kind!r}")
