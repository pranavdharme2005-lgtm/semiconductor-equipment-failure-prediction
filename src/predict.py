"""
Inference & Risk Scoring Engine with Explainable AI Integration.

Handles model loading, batch inference, real-time single-instance prediction,
failure probability calculation, risk categorization, and human-understandable XAI explanations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import joblib

from src.config import config
from src.schema import validate_schema


class FailurePredictor:
    """
    Inference & Explainability Engine for Semiconductor Equipment Failure Prediction.
    Calculates failure probabilities, maps them to risk categories (LOW, MEDIUM, HIGH),
    and generates automated operational alert messages and feature contributions.
    """

    def __init__(self, model_path: Optional[Path | str] = None, scaler_path: Optional[Path | str] = None):
        self.model_path = Path(model_path) if model_path else config.latest_model_path
        self.scaler_path = Path(scaler_path) if scaler_path else (config.model_dir / "scaler.joblib")
        
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.operating_threshold: float = 0.25
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads serialized model, scaler, and threshold config objects if available."""
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

        if self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)

        thresh_file = config.model_dir / "threshold_config.json"
        if thresh_file.exists():
            import json
            with open(thresh_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                self.operating_threshold = cfg.get("operating_threshold", 0.25)

    def calculate_risk_score(self, failure_probability: float) -> Tuple[str, str]:
        """
        Maps continuous failure probability to operational risk level using operating threshold (0.25).

        Args:
            failure_probability (float): Model output probability [0.0 - 1.0].

        Returns:
            Tuple[str, str]: (Risk Level Category, Actionable Recommendation)
        """
        low_t = self.operating_threshold * 0.70  # 0.18
        high_t = self.operating_threshold        # 0.25

        if failure_probability < low_t:
            return "LOW", "Equipment operating within normal parameters. Standard routine monitoring."
        elif failure_probability < high_t:
            return "MEDIUM", "Elevated sensor telemetry variance detected. Schedule routine maintenance check."
        else:
            return "HIGH", "CRITICAL RISK: Immediate equipment inspection required to prevent wafer scrap!"

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs batch failure risk inference on an equipment telemetry dataset.

        Args:
            df (pd.DataFrame): Processed feature DataFrame.

        Returns:
            pd.DataFrame: Original DataFrame augmented with failure_prob, predicted_failure, and risk_level.
        """
        if self.model is None or self.scaler is None:
            self._load_artifacts()

        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or Scaler artifact is not loaded or trained yet.")

        results_df = df.copy()
        exclude_cols = [config.target_column, config.id_column, config.timestamp_column]
        feature_cols = [col for col in results_df.columns if col not in exclude_cols]

        X = results_df[feature_cols]
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=feature_cols)

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[:, 1]
        else:
            probs = self.model.predict(X_scaled).astype(float)

        predictions = (probs >= self.operating_threshold).astype(int)

        results_df["failure_probability"] = probs
        results_df["predicted_failure"] = predictions
        
        risk_info = [self.calculate_risk_score(p) for p in probs]
        results_df["risk_level"] = [r[0] for r in risk_info]
        results_df["recommended_action"] = [r[1] for r in risk_info]

        return results_df

    def predict_single_with_explanation(self, sensor_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes single-instance prediction integrated with Explainable AI (XAI) feature analysis.

        Args:
            sensor_dict (Dict[str, Any]): Dictionary of sensor telemetry readings.

        Returns:
            Dict[str, Any]: Structured prediction payload with probabilities, risk level, and top risk factors.
        """
        from src.explainability import EquipmentExplainer

        explainer = EquipmentExplainer(model=self.model, scaler=self.scaler)
        return explainer.explain_instance(sensor_dict)


def build_predictor() -> FailurePredictor:
    """Returns a fresh FailurePredictor instance."""
    return FailurePredictor()
