"""
Comprehensive Unit Tests for Stage 3 Preprocessing Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import EquipmentDataPreprocessor


@pytest.fixture
def sample_telemetry_dataframe():
    """Generates sample unsorted telemetry data with missing values and duplicates."""
    return pd.DataFrame({
        "equipment_id": ["EQ_101", "EQ_101", "EQ_101", "EQ_102", "EQ_102"],
        "timestamp": pd.to_datetime([
            "2026-01-01 12:00:00",
            "2026-01-01 10:00:00",  # Unsorted
            "2026-01-01 10:00:00",  # Duplicate timestamp
            "2026-01-01 11:00:00",
            "2026-01-01 12:00:00"
        ]),
        "temperature": [75.5, 70.0, 70.0, np.nan, 85.0],
        "pressure": [12.1, 10.0, 10.0, 14.5, 15.0],
        "vibration": [0.85, 0.80, 0.80, 1.20, 1.30],
        "voltage": [220.0, 220.0, 220.0, 218.5, 218.0],
        "current": [15.2, 14.0, 14.0, 16.8, 17.0],
        "rpm": [3200.0, 3000.0, 3000.0, 3150.0, 3100.0],
        "flow_rate": [45.0, 40.0, 40.0, 48.2, 49.0],
        "runtime_hours": [1200.0, 1198.0, 1198.0, 1350.0, 1351.0],
        "maintenance_count": [3, 3, 3, 4, 4],
        "failure": [0, 0, 0, 0, 1]
    })


def test_parse_and_sort_chronologically(sample_telemetry_dataframe):
    """Tests chronological sorting per equipment_id."""
    preprocessor = EquipmentDataPreprocessor()
    sorted_df = preprocessor.parse_and_sort_chronologically(sample_telemetry_dataframe)
    
    eq101_ts = sorted_df[sorted_df["equipment_id"] == "EQ_101"]["timestamp"].tolist()
    assert eq101_ts == sorted(eq101_ts)


def test_deduplicate_records(sample_telemetry_dataframe):
    """Tests duplicate record detection and removal."""
    preprocessor = EquipmentDataPreprocessor()
    dedup_df, removed = preprocessor.deduplicate_records(sample_telemetry_dataframe)
    
    assert removed == 1
    assert len(dedup_df) == 4


def test_impute_missing_values(sample_telemetry_dataframe):
    """Tests group-wise time-series imputation leaving zero missing values."""
    preprocessor = EquipmentDataPreprocessor()
    sorted_df = preprocessor.parse_and_sort_chronologically(sample_telemetry_dataframe)
    dedup_df, _ = preprocessor.deduplicate_records(sorted_df)
    imputed_df = preprocessor.impute_missing_values(dedup_df)

    assert imputed_df.isnull().sum().sum() == 0
    # EQ_102 temperature at 11:00 should be bfilled from 12:00 (85.0)
    eq102_temp = imputed_df[imputed_df["equipment_id"] == "EQ_102"]["temperature"].iloc[0]
    assert eq102_temp == 85.0


def test_handle_outliers_clipping():
    """Tests that physical data errors (out of domain) are clipped while preserving valid signals."""
    df_outlier = pd.DataFrame({
        "equipment_id": ["EQ_101", "EQ_101"],
        "timestamp": pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 11:00:00"]),
        "temperature": [-10.0, 125.0],  # -10 is impossible physical error, 125 is degradation anomaly
        "pressure": [12.0, 12.0],
        "vibration": [0.8, 0.8],
        "voltage": [220.0, 220.0],
        "current": [15.0, 15.0],
        "rpm": [3000.0, 3000.0],
        "flow_rate": [45.0, 45.0],
        "runtime_hours": [100.0, 101.0],
        "maintenance_count": [1, 1],
        "failure": [0, 1]
    })

    preprocessor = EquipmentDataPreprocessor()
    cleaned_df, clip_counts = preprocessor.handle_outliers(df_outlier)

    # -10.0 clipped to 0.0 °C (schema minimum)
    assert cleaned_df["temperature"].iloc[0] == 0.0
    # 125.0 preserved intact as true anomaly (under max schema bound 150.0 °C)
    assert cleaned_df["temperature"].iloc[1] == 125.0
    assert clip_counts["temperature"] == 1


def test_categorical_encoding(sample_telemetry_dataframe):
    """Tests categorical encoding of equipment_id into equipment_code."""
    preprocessor = EquipmentDataPreprocessor()
    encoded_df = preprocessor.encode_categorical_features(sample_telemetry_dataframe)
    
    assert "equipment_code" in encoded_df.columns
    assert encoded_df["equipment_code"].dtype in [np.int32, np.int64, int]
