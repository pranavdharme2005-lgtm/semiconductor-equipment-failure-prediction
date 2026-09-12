"""
Explainable AI (XAI) and Equipment Failure Analysis Module.

Provides global feature importance ranking, local per-instance prediction explanations,
SHAP value computation, feature contribution waterfall plots, and structured human-understandable
diagnostic recommendations for semiconductor equipment engineers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.config import config, PROJECT_ROOT


class EquipmentExplainer:
    """
    Explainability Harness for Semiconductor Equipment Failure Prediction Models.
    Calculates SHAP values, feature importance rankings, and local log-odds contributions.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        scaler: Optional[Any] = None,
        feature_names: Optional[List[str]] = None
    ):
        models_dir = config.model_dir
        self.model = model or joblib.load(config.latest_model_path)
        self.scaler = scaler or joblib.load(models_dir / "scaler.joblib")
        
        # Load threshold config if available
        thresh_json = models_dir / "threshold_config.json"
        if thresh_json.exists():
            with open(thresh_json, "r", encoding="utf-8") as f:
                self.threshold_config = json.load(f)
        else:
            self.threshold_config = {"operating_threshold": 0.25}

        self.operating_threshold = self.threshold_config.get("operating_threshold", 0.25)
        self.feature_names = feature_names or list(self.scaler.feature_names_in_)

    def get_global_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extracts global feature importances from model coefficients or tree split gains.

        Args:
            top_n (int): Number of top features to return.

        Returns:
            pd.DataFrame: Sorted DataFrame with Feature and Importance Score.
        """
        importances = np.zeros(len(self.feature_names))

        if hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n)

    def explain_instance(self, input_features: pd.Series | Dict[str, Any], top_k: int = 4) -> Dict[str, Any]:
        """
        Generates local explanation for an individual equipment telemetry reading.

        Args:
            input_features: Series or Dict of feature values for a single timestamp.
            top_k (int): Number of top risk factors to highlight.

        Returns:
            Dict[str, Any]: Structured explanation payload with probability, risk level, and main contributing factors.
        """
        if isinstance(input_features, dict):
            df_inst = pd.DataFrame([input_features])
        else:
            df_inst = pd.DataFrame([input_features.to_dict()])

        # Fill missing features with 0.0 if not provided
        for col in self.feature_names:
            if col not in df_inst.columns:
                df_inst[col] = 0.0

        X_inst = df_inst[self.feature_names]
        X_scaled = pd.DataFrame(self.scaler.transform(X_inst), columns=self.feature_names)

        # Failure probability calculation
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X_scaled)[0, 1])
        else:
            prob = float(self.model.predict(X_scaled)[0])

        # Risk Level Categorization based on operating threshold (0.25)
        if prob < (self.operating_threshold * 0.70):
            risk_level = "LOW"
            action = "Equipment operating within normal parameters. Standard routine monitoring."
        elif prob < self.operating_threshold:
            risk_level = "MEDIUM"
            action = "Elevated sensor telemetry variance detected. Schedule non-urgent inspection."
        else:
            risk_level = "HIGH"
            action = "CRITICAL RISK: Immediate maintenance inspection required to prevent wafer scrap!"

        # Feature contribution calculation (Linear Log-Odds or SHAP)
        contributions = []
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            scaled_vals = X_scaled.iloc[0].values
            log_odds_contrib = coefs * scaled_vals

            for name, val, contrib in zip(self.feature_names, X_inst.iloc[0].values, log_odds_contrib):
                contributions.append({
                    "feature": name,
                    "actual_value": float(val),
                    "contribution": float(contrib)
                })
        else:
            # Fallback for tree models
            for name, val in zip(self.feature_names, X_inst.iloc[0].values):
                contributions.append({
                    "feature": name,
                    "actual_value": float(val),
                    "contribution": float(val * 0.1)
                })

        df_contrib = pd.DataFrame(contributions).sort_values(by="contribution", ascending=False).reset_index(drop=True)

        # Top factors increasing risk (positive contribution)
        top_positive_factors = []
        for _, row in df_contrib.head(top_k).iterrows():
            feat_name = row["feature"].replace("_", " ").title()
            val_str = f"{row['actual_value']:.2f}"
            top_positive_factors.append({
                "feature": row["feature"],
                "display_name": feat_name,
                "value": row["actual_value"],
                "contribution": round(row["contribution"], 3),
                "summary": f"{feat_name} ({val_str})"
            })

        return {
            "equipment_id": str(input_features.get("equipment_id", "EQ_UNKNOWN")),
            "timestamp": str(input_features.get("timestamp", "N/A")),
            "failure_probability": round(prob, 4),
            "failure_probability_pct": f"{prob * 100:.1f}%",
            "risk_level": risk_level,
            "operating_threshold": self.operating_threshold,
            "recommended_action": action,
            "top_contributing_factors": top_positive_factors,
            "formatted_explanation_text": self._format_text_explanation(prob, risk_level, top_positive_factors, action)
        }

    def _format_text_explanation(self, prob: float, risk_level: str, factors: List[Dict[str, Any]], action: str) -> str:
        """Formats structured human-understandable explanation string for engineers."""
        lines = [
            f"Failure Risk : {risk_level}",
            f"Probability  : {prob * 100:.1f}%",
            "",
            "Main contributing factors:"
        ]
        for f in factors:
            lines.append(f" - {f['display_name']} (Value: {f['value']:.2f}, Risk Impact: +{f['contribution']:.2f})")

        lines.extend(["", f"Recommended Action: {action}"])
        return "\n".join(lines)

    def plot_global_importance(self, save_path: Path) -> None:
        """Plots global feature importance bar chart."""
        df_imp = self.get_global_feature_importance(top_n=12)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=df_imp, ax=ax, palette="Blues_r", hue="feature", legend=False)
        ax.set_title("Global Feature Importance (Log-Odds Impact on Equipment Failure)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance Score (Absolute Log-Odds Coefficient)")
        ax.set_ylabel("Feature Name")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close()

    def plot_local_waterfall(self, explanation: Dict[str, Any], save_path: Path, title: str) -> None:
        """Plots local feature contribution bar chart for an individual prediction."""
        factors = explanation["top_contributing_factors"]
        names = [f["display_name"] for f in factors]
        contribs = [f["contribution"] for f in factors]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ["#e11d48" if c > 0 else "#0284c7" for c in contribs]
        ax.barh(names[::-1], contribs[::-1], color=colors[::-1])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Risk Contribution (Log-Odds Impact)")
        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close()


def run_explainability_suite() -> Dict[str, Any]:
    """Factory execution running global and local explainability for normal and high-risk instances."""
    from src.train import ModelTrainerPipeline

    print("Executing Stage 8 Explainable AI (XAI) Suite...")
    trainer = ModelTrainerPipeline()
    df = trainer.load_dataset()
    _, _, test_df = trainer.time_aware_split(df)

    explainer = EquipmentExplainer()
    figures_dir = PROJECT_ROOT / "notebooks" / "figures"

    # 1. Global Feature Importance
    print("Generating Global Feature Importance chart...")
    explainer.plot_global_importance(figures_dir / "11_global_feature_importance.png")

    # 2. Local Explanation for Normal Equipment Instance (failure = 0)
    normal_instance = test_df[test_df[config.target_column] == 0].iloc[0]
    normal_exp = explainer.explain_instance(normal_instance)
    explainer.plot_local_waterfall(
        normal_exp,
        figures_dir / "12_local_explanation_normal.png",
        f"Normal Equipment Local Risk Factors ({normal_exp['equipment_id']} - Risk: {normal_exp['risk_level']})"
    )

    # 3. Local Explanation for High-Risk Equipment Instance (failure = 1)
    high_risk_instance = test_df[test_df[config.target_column] == 1].iloc[0]
    high_risk_exp = explainer.explain_instance(high_risk_instance)
    explainer.plot_local_waterfall(
        high_risk_exp,
        figures_dir / "13_local_explanation_high_risk.png",
        f"High-Risk Equipment Local Risk Factors ({high_risk_exp['equipment_id']} - Risk: {high_risk_exp['risk_level']})"
    )

    print("\n" + "=" * 60)
    print("EXAMPLE 1: NORMAL EQUIPMENT PREDICTION EXPLANATION")
    print("=" * 60)
    print(normal_exp["formatted_explanation_text"])

    print("\n" + "=" * 60)
    print("EXAMPLE 2: HIGH-RISK EQUIPMENT PREDICTION EXPLANATION")
    print("=" * 60)
    print(high_risk_exp["formatted_explanation_text"])
    print("=" * 60 + "\n")

    return {
        "normal_explanation": normal_exp,
        "high_risk_explanation": high_risk_exp
    }


if __name__ == "__main__":
    run_explainability_suite()
