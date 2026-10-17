from __future__ import annotations
import inspect
from typing import Any
import numpy as np
from sklearn.neural_network import MLPClassifier


def _filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid}


class MLPModel:
    """Deep Multilayer Perceptron wrapper with sample-weight rebalancing support."""

    def __init__(self, params: dict[str, Any] | None = None):
        filtered = _filter_kwargs(MLPClassifier, params or {})
        self._model = MLPClassifier(**filtered)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> MLPModel:
        if sample_weight is not None:
            # Resample dataset based on sample_weight distribution since MLPClassifier lacks native sample_weight
            rng = np.random.default_rng(getattr(self._model, "random_state", 42) or 42)
            p = sample_weight / np.sum(sample_weight)
            indices = rng.choice(len(X), size=len(X), replace=True, p=p)
            X = X[indices]
            y = y[indices]

        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        if hasattr(self._model, "coefs_") and self._model.coefs_:
            # Average input-layer absolute weights across first hidden layer neurons
            return np.abs(self._model.coefs_[0]).mean(axis=1)
        return np.zeros(getattr(self._model, "n_features_in_", 0))


def mlp_classifier(params: dict[str, Any] | None = None) -> MLPModel:
    return MLPModel(params)
