"""
Unit Tests for Explainable AI (XAI) and Failure Risk Explainer.
"""

import pytest
import pandas as pd
from src.explainability import EquipmentExplainer


@pytest.fixture
def sample_features_row():
    return {
        "equipment_id": "EQ_101",
        "timestamp": "2026-01-01 10:00:00",
        "temperature": 98.5,
        "pressure": 8.5,
        "vibration": 4.1,
        "voltage": 220.0,
        "current": 28.5,
        "rpm": 2720.0,
        "flow_rate": 35.0,
        "runtime_hours": 1500.0,
        "maintenance_count": 2,
        "equipment_code": 0,
        "vib_roll_mean_6h": 3.9,
        "temp_roll_mean_6h": 96.0,
        "thermal_strain_index": 403.85
    }


def test_global_feature_importance():
    """Verifies global feature importance extraction."""
    explainer = EquipmentExplainer()
    df_imp = explainer.get_global_feature_importance(top_n=10)

    assert len(df_imp) == 10
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert df_imp["importance"].iloc[0] >= df_imp["importance"].iloc[-1]


def test_local_instance_explanation(sample_features_row):
    """Verifies per-instance prediction explanation payload structure."""
    explainer = EquipmentExplainer()
    exp = explainer.explain_instance(sample_features_row)

    assert "failure_probability" in exp
    assert "risk_level" in exp
    assert "top_contributing_factors" in exp
    assert "formatted_explanation_text" in exp
    assert exp["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(exp["top_contributing_factors"]) > 0
