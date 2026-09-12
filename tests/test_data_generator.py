"""
Unit Tests for Data Generator and Data Validator Modules.
"""

import pytest
import pandas as pd
from src.data_generator import generate_equipment_telemetry
from src.validate_data import DatasetValidator


def test_generate_equipment_telemetry_shape():
    """Verifies shape and record count of generated telemetry data."""
    df = generate_equipment_telemetry(num_equipment=3, records_per_equipment=100, random_state=42)
    assert len(df) == 300
    assert "equipment_id" in df.columns
    assert "failure" in df.columns
    assert df["equipment_id"].nunique() == 3


def test_data_validator_checks():
    """Verifies that DatasetValidator correctly identifies shape, missing values, and target distribution."""
    df = generate_equipment_telemetry(num_equipment=2, records_per_equipment=50, random_state=42)
    validator = DatasetValidator(df)

    missing_df = validator.check_missing_values()
    target_dist = validator.check_target_distribution()
    duplicates = validator.check_duplicates()

    assert "Missing Count" in missing_df.columns
    assert "failure_rate_pct" in target_dist
    assert duplicates["total_duplicate_rows"] == 0
