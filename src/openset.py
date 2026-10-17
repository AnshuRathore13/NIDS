import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def unknown_auroc(closed_probs: np.ndarray, unknown_probs: np.ndarray) -> float:
    """AUROC for detecting unknown attacks using prediction confidence."""
    closed_conf = np.max(closed_probs, axis=1)
    unknown_conf = np.max(unknown_probs, axis=1)
    y_true = np.concatenate([np.zeros(len(closed_conf)), np.ones(len(unknown_conf))])
    y_scores = np.concatenate([1.0 - closed_conf, 1.0 - unknown_conf])
    return float(roc_auc_score(y_true, y_scores))


def threshold_sweep(closed_probs: np.ndarray, unknown_probs: np.ndarray, thresholds: list[float] | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = [round(float(t), 2) for t in np.arange(0.05, 1.0, 0.05)]

    closed_conf = np.max(closed_probs, axis=1)
    unknown_conf = np.max(unknown_probs, axis=1)

    records = []
    for t in thresholds:
        detection_rate = float(np.mean(unknown_conf < t))
        false_alarm_rate = float(np.mean(closed_conf < t))
        records.append({
            "threshold": t,
            "detection_rate": round(detection_rate, 4),
            "false_alarm_rate": round(false_alarm_rate, 4)
        })
    return pd.DataFrame(records)
