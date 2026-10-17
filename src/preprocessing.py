from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

CATEGORICAL_FEATURES = {"proto", "service", "state"}
ENGINEERED_FEATURES = {"total_bytes", "total_pkts", "byte_ratio", "pkt_ratio", "avg_pkt_size", "duration_log"}


def balanced_sample_weight(y: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Square-root balanced class weights to address minority attack classes without precision collapse."""
    return compute_sample_weight("balanced", y) ** power


class Preprocessor:
    def __init__(self, feature_list: list[str] | None = None):
        self.feature_list = feature_list
        self.scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.target_encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame, target_col: str = "label_merged") -> Preprocessor:
        df = df.copy()
        if self.feature_list is None:
            self.feature_list = [c for c in df.columns if c not in {target_col, "id", "attack_cat"}]

        # Categorical features
        for col in self.feature_list:
            if col in CATEGORICAL_FEATURES and col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le

        # Numeric features
        numeric_cols = [c for c in self.feature_list if c not in CATEGORICAL_FEATURES and c in df.columns]
        if numeric_cols:
            self.scaler.fit(df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0))

        if target_col in df.columns:
            self.target_encoder.fit(df[target_col].astype(str))

        return self

    def transform(self, df: pd.DataFrame, target_col: str = "label_merged") -> tuple[np.ndarray, np.ndarray | None]:
        df = df.copy()
        processed = []
        for col in self.feature_list:
            if col in self.label_encoders:
                classes = set(self.label_encoders[col].classes_)
                safe_col = df[col].astype(str).map(lambda x: x if x in classes else list(classes)[0])
                processed.append(self.label_encoders[col].transform(safe_col).reshape(-1, 1))
            elif col in df.columns:
                val = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.reshape(-1, 1)
                processed.append(val)

        X = np.hstack(processed) if processed else np.empty((len(df), 0))
        y = self.target_encoder.transform(df[target_col].astype(str)) if target_col in df.columns else None
        return X, y
