"""
Unit Tests for Data Schema and Validation Rules.
"""

import pytest
import pandas as pd
import numpy as np
from src.schema import validate_schema, EXPECTED_COLUMN_NAMES, TARGET_COLUMN


@pytest.fixture
def valid_sample_dataframe():
    """Generates a valid sample DataFrame conforming to expected schema."""
    return pd.DataFrame({
        "equipment_id": ["EQ_101", "EQ_102"],
        "timestamp": pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 11:00:00"]),
        "temperature": [75.5, 82.0],
        "pressure": [12.1, 14.5],
        "vibration": [0.85, 1.20],
        "voltage": [220.0, 218.5],
        "current": [15.2, 16.8],
        "rpm": [3200.0, 3150.0],
        "flow_rate": [45.0, 48.2],
        "runtime_hours": [1200.0, 1350.0],
        "maintenance_count": [3, 4],
        "failure": [0, 1]
    })


def test_schema_expected_columns():
    """Verifies that expected column list contains all required features."""
    expected = [
        "equipment_id", "timestamp", "temperature", "pressure", "vibration",
        "voltage", "current", "rpm", "flow_rate", "runtime_hours",
        "maintenance_count", "failure"
    ]
    assert sorted(EXPECTED_COLUMN_NAMES) == sorted(expected)


def test_validate_schema_valid_data(valid_sample_dataframe):
    """Tests validation success on valid sample data."""
    is_valid, errors = validate_schema(valid_sample_dataframe)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_schema_missing_column(valid_sample_dataframe):
    """Tests validation failure when a required column is missing."""
    invalid_df = valid_sample_dataframe.drop(columns=["vibration"])
    is_valid, errors = validate_schema(invalid_df)
    assert is_valid is False
    assert any("vibration" in err for err in errors)


def test_validate_schema_out_of_bounds(valid_sample_dataframe):
    """Tests validation failure when numerical values exceed permissible limits."""
    invalid_df = valid_sample_dataframe.copy()
    invalid_df.loc[0, "temperature"] = 999.0  # Max is 150.0 °C
    is_valid, errors = validate_schema(invalid_df)
    assert is_valid is False
    assert any("temperature" in err for err in errors)
