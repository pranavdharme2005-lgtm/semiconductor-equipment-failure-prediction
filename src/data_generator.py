"""
Synthetic Data Generation Module for Semiconductor Equipment Telemetry.

Generates realistic time-series sensor data mimicking semiconductor wafer processing tools
(e.g., Plasma Etch, CVD chambers, CMP tools) with physics-inspired degradation trends,
stochastic failure events, realistic missing values, and sensor noise.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from src.config import config


def generate_equipment_telemetry(
    num_equipment: int = 10,
    records_per_equipment: int = 1000,
    start_date: str = "2026-01-01 00:00:00",
    freq: str = "1h",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic semiconductor equipment telemetry dataset.

    Args:
        num_equipment (int): Number of distinct equipment units (e.g. 10 -> EQ_101 to EQ_110).
        records_per_equipment (int): Number of timestamp readings per equipment unit.
        start_date (str): Start date string for datetime generation.
        freq (str): Frequency of telemetry collection (e.g. '1h').
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Generated dataset containing raw sensor features, operational metrics, and failure labels.
    """
    np.random.seed(random_state)
    records = []

    equipment_ids = [f"EQ_{101 + i}" for i in range(num_equipment)]
    timestamps = pd.date_range(start=start_date, periods=records_per_equipment, freq=freq)

    for eq_id in equipment_ids:
        # Initial baseline states for each equipment unit
        runtime_hours = np.random.uniform(100, 2500)
        maintenance_count = np.random.randint(0, 5)
        
        # State tracking: 0 = NORMAL, >0 = hours remaining in DEGRADING state
        degradation_counter = 0
        hours_since_last_maint = 0

        # Base physical setpoints with slight unit-to-unit calibration variance
        base_temp = 75.0 + np.random.uniform(-3, 3)
        base_press = 12.0 + np.random.uniform(-0.5, 0.5)
        base_vib = 0.8 + np.random.uniform(-0.1, 0.1)
        base_volt = 220.0 + np.random.uniform(-2, 2)
        base_curr = 15.0 + np.random.uniform(-1, 1)
        base_rpm = 3200.0 + np.random.uniform(-50, 50)
        base_flow = 45.0 + np.random.uniform(-2, 2)

        for ts in timestamps:
            runtime_hours += 1.0
            hours_since_last_maint += 1

            # Cumulative wear hazard increases with runtime since last maintenance
            wear_factor = 1.0 + (hours_since_last_maint / 1500.0)

            # Trigger degradation phase stochastically if normal
            if degradation_counter == 0:
                # Hazard rate increases with wear_factor
                trigger_prob = 0.035 * wear_factor
                if np.random.rand() < trigger_prob:
                    degradation_counter = np.random.randint(8, 20)  # Degrading phase lasts 8-20 hours

            # Determine operational state & failure label
            failure = 0
            if degradation_counter > 0:
                degradation_counter -= 1
                # Progression intensity increases as counter approaches 0
                progression = (36 - degradation_counter) / 36.0

                # Sensor drifts during degradation
                temp = base_temp + (25.0 * progression) + np.random.normal(0, 2.5)
                press = base_press + (np.random.choice([-1, 1]) * 4.5 * progression) + np.random.normal(0, 1.2)
                vib = base_vib + (3.2 * progression) + np.random.normal(0, 0.3)
                volt = base_volt + np.random.normal(0, 3.5)
                curr = base_curr + (12.0 * progression) + np.random.normal(0, 1.8)
                rpm = base_rpm - (450.0 * progression) + np.random.normal(0, 40)
                flow = base_flow - (8.0 * progression) + np.random.normal(0, 1.5)

                # Final timestamp of degradation phase represents failure event
                if degradation_counter == 0:
                    failure = 1
                    # Perform maintenance after failure
                    maintenance_count += 1
                    hours_since_last_maint = 0
            else:
                # Normal operational noise
                temp = base_temp + np.random.normal(0, 1.2)
                press = base_press + np.random.normal(0, 0.4)
                vib = base_vib + np.random.normal(0, 0.08)
                volt = base_volt + np.random.normal(0, 1.5)
                curr = base_curr + np.random.normal(0, 0.8)
                rpm = base_rpm + np.random.normal(0, 25)
                flow = base_flow + np.random.normal(0, 1.0)

            records.append({
                "equipment_id": eq_id,
                "timestamp": ts,
                "temperature": float(np.round(temp, 2)),
                "pressure": float(np.round(press, 2)),
                "vibration": float(np.round(vib, 3)),
                "voltage": float(np.round(volt, 2)),
                "current": float(np.round(curr, 2)),
                "rpm": float(np.round(rpm, 1)),
                "flow_rate": float(np.round(flow, 2)),
                "runtime_hours": float(np.round(runtime_hours, 1)),
                "maintenance_count": int(maintenance_count),
                "failure": int(failure)
            })

    df = pd.DataFrame(records)

    # Inject realistic missing values (~1.5% random NaNs across sensor columns)
    sensor_cols = ["temperature", "pressure", "vibration", "voltage", "current", "rpm", "flow_rate"]
    for col in sensor_cols:
        mask = np.random.rand(len(df)) < 0.015
        df.loc[mask, col] = np.nan

    # Inject a small number of sensor outliers (~0.3% extreme spikes due to electrical/noise glitches)
    for col in ["vibration", "temperature", "pressure"]:
        spike_mask = np.random.rand(len(df)) < 0.003
        if col == "vibration":
            df.loc[spike_mask, col] = np.random.uniform(8.0, 9.8)
        elif col == "temperature":
            df.loc[spike_mask, col] = np.random.uniform(135.0, 148.0)
        elif col == "pressure":
            df.loc[spike_mask, col] = np.random.uniform(21.0, 24.5)

    return df


def generate_and_save_dataset(
    output_path: Optional[Path | str] = None,
    num_equipment: int = 10,
    records_per_equipment: int = 1000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Path]:
    """
    Generates and saves raw equipment telemetry dataset to disk.

    Args:
        output_path (Optional[Path | str]): Destination CSV path. Defaults to config raw_data_path.
        num_equipment (int): Equipment unit count.
        records_per_equipment (int): Records per equipment.
        random_state (int): Seed.

    Returns:
        Tuple[pd.DataFrame, Path]: (Generated DataFrame, Saved File Path)
    """
    target_path = Path(output_path) if output_path else config.raw_data_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic telemetry data for {num_equipment} equipment units ({records_per_equipment} records each)...")
    df = generate_equipment_telemetry(
        num_equipment=num_equipment,
        records_per_equipment=records_per_equipment,
        random_state=random_state
    )

    df.to_csv(target_path, index=False)
    print(f"Successfully generated {len(df)} records and saved to {target_path}")

    return df, target_path


if __name__ == "__main__":
    generate_and_save_dataset()
