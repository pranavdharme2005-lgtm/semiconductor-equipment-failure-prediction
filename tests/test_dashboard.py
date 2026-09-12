"""
Unit and Integration Tests for Streamlit Dashboard Data Engine.
"""

import pytest
import pandas as pd
from dashboard.app import load_feature_dataset, load_prediction_engine


def test_dashboard_dataset_loading():
    """Verifies that dashboard data loader retrieves feature dataset cleanly."""
    df = load_feature_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "equipment_id" in df.columns
    assert "failure" in df.columns


def test_dashboard_prediction_engine():
    """Verifies that dashboard prediction engine initializes and runs inference."""
    predictor = load_prediction_engine()
    assert predictor is not None
    assert predictor.model is not None
    assert predictor.scaler is not None
