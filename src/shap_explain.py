import numpy as np
import pandas as pd
import shap


def compute_shap_importance(model, background_data: np.ndarray, eval_data: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    shap_values = explainer.shap_values(eval_data)
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    df = pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)
