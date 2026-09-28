"""Explainable AI (TreeSHAP) Engine for AeroFuel AI.

Provides global feature attribution, single-flight prediction explanations (waterfall),
and partial dependence analysis using Shapley Additive Explanations (SHAP).
Adheres strictly to the operational interpretation guideline:
"The model associates these variables with the prediction" (never claiming direct physical causality).
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger("ShapExplainer")


class AeroFuelExplainer:
    """Wrapper around TreeSHAP for tree-based flight fuel prediction models."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        # If the model is a wrapper like PhysicsResidualHybrid, use its inner booster
        if hasattr(model, "ml_booster"):
            self.explainer = shap.TreeExplainer(model.ml_booster)
        else:
            self.explainer = shap.TreeExplainer(model)

    def explain_global(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute mean absolute SHAP value across all features."""
        logger.info("Computing global SHAP values for %d samples...", len(X))
        shap_values = self.explainer.shap_values(X[self.feature_names])
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        df_importance = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP_kg": mean_abs_shap,
        }).sort_values(by="Mean_Abs_SHAP_kg", ascending=False)

        total_impact = df_importance["Mean_Abs_SHAP_kg"].sum()
        df_importance["Relative_Importance_pct"] = (df_importance["Mean_Abs_SHAP_kg"] / max(total_impact, 1e-6)) * 100.0

        return df_importance

    def explain_local_sample(self, sample: pd.Series, top_k: int = 6) -> Dict[str, Any]:
        """Explain a single flight prediction by returning top associated drivers."""
        sample_df = pd.DataFrame([sample])[self.feature_names]
        shap_vals = self.explainer.shap_values(sample_df)[0]
        base_val = float(self.explainer.expected_value)

        drivers = []
        for i, feat in enumerate(self.feature_names):
            impact = float(shap_vals[i])
            val = float(sample_df.iloc[0, i])
            drivers.append({
                "feature": feat,
                "feature_value": round(val, 2),
                "shap_impact_kg": round(impact, 2),
                "direction": "Increases Fuel" if impact > 0 else "Decreases Fuel",
            })

        # Sort by absolute SHAP impact
        drivers.sort(key=lambda x: abs(x["shap_impact_kg"]), reverse=True)

        return {
            "base_prediction_kg": round(base_val, 2),
            "total_shap_adjustment_kg": round(float(np.sum(shap_vals)), 2),
            "top_drivers": drivers[:top_k],
            "association_statement": (
                "Under the trained model, the features above are associated with the predicted delta "
                "relative to the training population mean. Note: SHAP indicates statistical association within "
                "the model's learned representation, not physical causation."
            ),
        }
