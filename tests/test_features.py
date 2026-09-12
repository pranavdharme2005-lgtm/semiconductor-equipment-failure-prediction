"""
Comprehensive Unit Tests for Stage 5 Feature Engineering Module & Data Leakage Auditing.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import EquipmentFeatureEngineer


@pytest.fixture
def sample_telemetry_series():
    """Generates a 10-hour telemetry series for EQ_101."""
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1h")
    return pd.DataFrame({
        "equipment_id": ["EQ_101"] * 10,
        "timestamp": timestamps,
        "temperature": [70.0, 71.0, 72.5, 75.0, 80.0, 88.0, 95.0, 105.0, 115.0, 125.0],
        "pressure": [12.0, 12.1, 11.9, 11.5, 10.8, 9.5, 8.2, 7.0, 6.1, 5.0],
        "vibration": [0.8, 0.82, 0.85, 1.1, 1.5, 2.2, 3.1, 4.2, 5.5, 6.2],
        "voltage": [220.0, 219.5, 220.5, 218.0, 219.0, 221.0, 218.5, 219.0, 220.0, 218.0],
        "current": [15.0, 15.2, 15.5, 17.0, 19.5, 23.0, 27.0, 31.0, 35.0, 39.0],
        "rpm": [3200.0, 3190.0, 3180.0, 3120.0, 3050.0, 2920.0, 2800.0, 2680.0, 2550.0, 2400.0],
        "flow_rate": [45.0, 44.8, 44.5, 43.0, 41.0, 38.5, 36.0, 33.0, 30.0, 27.0],
        "runtime_hours": [100.0 + i for i in range(10)],
        "maintenance_count": [2] * 10,
        "failure": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    })


def test_engineered_features_creation(sample_telemetry_series):
    """Tests that all required engineered features are created."""
    fe = EquipmentFeatureEngineer()
    df_feat = fe.transform(sample_telemetry_series)

    required_features = [
        "temp_change_1h", "press_change_1h", "vib_change_1h", "current_change_1h",
        "vib_roll_mean_3h", "vib_roll_mean_6h", "vib_roll_std_3h", "vib_roll_std_6h",
        "temp_roll_mean_3h", "temp_roll_mean_6h", "temp_roll_std_3h", "temp_roll_std_6h",
        "temp_trend_3h_6h", "vib_trend_3h_6h", "voltage_dev", "rpm_dev", "flow_rate_dev",
        "power_watts", "thermal_strain_index", "runtime_since_maint", "maintenance_freq",
        "vib_expanding_mean", "temp_expanding_mean", "vib_expanding_ratio"
    ]

    for feat in required_features:
        assert feat in df_feat.columns, f"Missing required engineered feature: {feat}"


def test_data_leakage_prevention(sample_telemetry_series):
    """
    CRITICAL LEAKAGE TEST:
    Verifies that engineered feature values at timestamp t=5 DO NOT CHANGE when future records (t=6..9)
    are truncated or modified!
    """
    fe = EquipmentFeatureEngineer()

    # 1. Transform full dataset (10 hours)
    df_full_feat = fe.transform(sample_telemetry_series)
    t5_full_values = df_full_feat.iloc[5].to_dict()

    # 2. Transform truncated dataset containing ONLY records up to t=5 (first 6 rows)
    sample_truncated = sample_telemetry_series.iloc[:6].copy()
    df_trunc_feat = fe.transform(sample_truncated)
    t5_trunc_values = df_trunc_feat.iloc[5].to_dict()

    # 3. Compare all engineered features at index 5
    engineered_cols = [col for col in df_full_feat.columns if col not in ["equipment_id", "timestamp", "failure"]]
    
    for col in engineered_cols:
        val_full = t5_full_values[col]
        val_trunc = t5_trunc_values[col]
        
        assert pytest.approx(val_full, abs=1e-5) == val_trunc, (
            f"DATA LEAKAGE DETECTED in feature '{col}'! "
            f"Value at index 5 changed when future records were included (Full={val_full}, Truncated={val_trunc})"
        )


def test_rate_of_change_calculations(sample_telemetry_series):
    """Tests 1-hour rate of change lag differences."""
    fe = EquipmentFeatureEngineer()
    df_feat = fe.transform(sample_telemetry_series)

    # At row 1: temp is 71.0, row 0 temp was 70.0 -> diff is 1.0
    assert pytest.approx(df_feat.loc[1, "temp_change_1h"], abs=1e-5) == 1.0
    # At row 1: current is 15.2, row 0 current was 15.0 -> diff is 0.2
    assert pytest.approx(df_feat.loc[1, "current_change_1h"], abs=1e-5) == 0.2


def test_nominal_deviations(sample_telemetry_series):
    """Tests nominal setpoint deviation calculations."""
    fe = EquipmentFeatureEngineer()
    df_feat = fe.transform(sample_telemetry_series)

    # Row 0 voltage is 220.0 -> voltage_dev = 0.0
    assert pytest.approx(df_feat.loc[0, "voltage_dev"], abs=1e-5) == 0.0
    # Row 0 rpm is 3200.0 -> rpm_dev = 0.0
    assert pytest.approx(df_feat.loc[0, "rpm_dev"], abs=1e-5) == 0.0
    # Row 0 flow_rate is 45.0 -> flow_rate_dev = 0.0
    assert pytest.approx(df_feat.loc[0, "flow_rate_dev"], abs=1e-5) == 0.0
