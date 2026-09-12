"""
Dataset Schema Definition and Data Validation Module.

Defines expected columns, data types, physical units, permissible numerical ranges,
and schema verification routines for Semiconductor Equipment Failure prediction data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ColumnDefinition:
    """Dataclass defining properties of a single dataset column."""
    name: str
    dtype: str
    is_target: bool = False
    unit: str = ""
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None


# Expected Dataset Schema Specification
EXPECTED_SCHEMA: List[ColumnDefinition] = [
    ColumnDefinition(
        name="equipment_id",
        dtype="object",
        unit="Identifier",
        description="Unique identifier of the semiconductor processing equipment (e.g. EQ_101)"
    ),
    ColumnDefinition(
        name="timestamp",
        dtype="datetime64[ns]",
        unit="ISO-8601",
        description="Date and timestamp of sensor telemetry reading"
    ),
    ColumnDefinition(
        name="temperature",
        dtype="float64",
        unit="°C",
        description="Wafer process chamber temperature",
        min_value=0.0,
        max_value=150.0
    ),
    ColumnDefinition(
        name="pressure",
        dtype="float64",
        unit="PSI",
        description="Vacuum chamber or liquid line pressure",
        min_value=0.0,
        max_value=25.0
    ),
    ColumnDefinition(
        name="vibration",
        dtype="float64",
        unit="mm/s",
        description="Mechanical component vibration magnitude",
        min_value=0.0,
        max_value=10.0
    ),
    ColumnDefinition(
        name="voltage",
        dtype="float64",
        unit="V",
        description="Equipment power supply input voltage",
        min_value=50.0,
        max_value=300.0
    ),
    ColumnDefinition(
        name="current",
        dtype="float64",
        unit="A",
        description="Electrical current draw during operation",
        min_value=0.0,
        max_value=100.0
    ),
    ColumnDefinition(
        name="rpm",
        dtype="float64",
        unit="RPM",
        description="Vacuum pump or motor rotational speed",
        min_value=0.0,
        max_value=10000.0
    ),
    ColumnDefinition(
        name="flow_rate",
        dtype="float64",
        unit="L/min",
        description="Process gas or coolant liquid flow rate",
        min_value=0.0,
        max_value=200.0
    ),
    ColumnDefinition(
        name="runtime_hours",
        dtype="float64",
        unit="hours",
        description="Cumulative operating hours since deployment",
        min_value=0.0,
        max_value=50000.0
    ),
    ColumnDefinition(
        name="maintenance_count",
        dtype="int64",
        unit="count",
        description="Total maintenance events recorded",
        min_value=0,
        max_value=100
    ),
    ColumnDefinition(
        name="failure",
        dtype="int64",
        is_target=True,
        unit="binary",
        description="Binary failure target flag (0 = Normal, 1 = Equipment Failure)",
        min_value=0,
        max_value=1
    ),
]

EXPECTED_COLUMN_NAMES = [col.name for col in EXPECTED_SCHEMA]
TARGET_COLUMN = "failure"
FEATURE_COLUMNS = [col.name for col in EXPECTED_SCHEMA if not col.is_target and col.name not in ["equipment_id", "timestamp"]]


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates whether an input DataFrame conforms to the expected schema.

    Args:
        df (pd.DataFrame): Input equipment telemetry dataset.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    if df.empty:
        return False, ["Input DataFrame is empty."]

    # Check for missing required columns
    missing_cols = set(EXPECTED_COLUMN_NAMES) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(list(missing_cols))}")

    # Return early if core columns missing
    if errors:
        return False, errors

    # Range & Domain Checks for numerical features
    for col_def in EXPECTED_SCHEMA:
        col = col_def.name
        if col not in df.columns:
            continue

        if col_def.min_value is not None:
            min_val = df[col].min()
            if min_val < col_def.min_value:
                errors.append(f"Column '{col}' has values ({min_val}) below minimum allowed ({col_def.min_value}).")

        if col_def.max_value is not None:
            max_val = df[col].max()
            if max_val > col_def.max_value:
                errors.append(f"Column '{col}' has values ({max_val}) above maximum allowed ({col_def.max_value}).")

    # Verify target column binary bounds if present
    if TARGET_COLUMN in df.columns:
        unique_targets = df[TARGET_COLUMN].dropna().unique()
        invalid_targets = [val for val in unique_targets if val not in [0, 1]]
        if invalid_targets:
            errors.append(f"Target column '{TARGET_COLUMN}' contains non-binary values: {invalid_targets}")

    is_valid = len(errors) == 0
    return is_valid, errors


def get_schema_summary() -> pd.DataFrame:
    """Returns a pandas DataFrame summary of the schema definition."""
    records = []
    for col in EXPECTED_SCHEMA:
        records.append({
            "Column Name": col.name,
            "Data Type": col.dtype,
            "Unit": col.unit,
            "Min Range": col.min_value if col.min_value is not None else "N/A",
            "Max Range": col.max_value if col.max_value is not None else "N/A",
            "Is Target": col.is_target,
            "Description": col.description
        })
    return pd.DataFrame(records)
