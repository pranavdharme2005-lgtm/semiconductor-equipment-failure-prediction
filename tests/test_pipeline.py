"""
Integration & End-to-End Pipeline Contract Tests.
"""

import pytest
from src.config import config
from src.predict import FailurePredictor
from src.evaluation import ModelEvaluator


def test_config_loader():
    """Verifies that configuration resolves paths and schema correctly."""
    assert config.id_column == "equipment_id"
    assert config.target_column == "failure"
    assert "temperature" in config.sensor_columns
    assert len(config.feature_columns) > 5


def test_risk_scoring():
    """Tests failure probability to risk score mapping using 0.25 operating threshold."""
    predictor = FailurePredictor()
    
    risk_low, _ = predictor.calculate_risk_score(0.10)
    risk_med, _ = predictor.calculate_risk_score(0.20)
    risk_high, _ = predictor.calculate_risk_score(0.85)

    assert risk_low == "LOW"
    assert risk_med == "MEDIUM"
    assert risk_high == "HIGH"


def test_evaluator_metrics():
    """Tests metric calculation helper with synthetic arrays."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 0]
    y_prob = [0.1, 0.2, 0.9, 0.4]

    metrics = ModelEvaluator.calculate_metrics(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 0.75
