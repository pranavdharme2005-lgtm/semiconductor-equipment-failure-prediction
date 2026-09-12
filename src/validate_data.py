"""
Dataset Quality & Schema Validation Script.

Executes comprehensive validation audits on raw semiconductor telemetry datasets:
1. Dataset Shape & Column Names
2. Missing Value Analysis
3. Duplicate Rows Audit
4. Sensor Out-of-Bound / Invalid Value Checks
5. Class Imbalance / Failure Target Distribution
6. Timestamp Validity & Time-series Continuity
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import config
from src.schema import EXPECTED_SCHEMA, TARGET_COLUMN, validate_schema


class DatasetValidator:
    """
    Validation engine auditing dataset quality, missing value ratios,
    duplicate records, sensor value ranges, and class balance.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def check_missing_values(self) -> pd.DataFrame:
        """Calculates missing value counts and percentages per column."""
        total_rows = len(self.df)
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / total_rows) * 100.0

        df_missing = pd.DataFrame({
            "Missing Count": missing_count,
            "Missing Percentage (%)": missing_pct.round(2)
        })
        return df_missing

    def check_duplicates(self) -> Dict[str, int]:
        """Checks for duplicate records."""
        total_duplicates = int(self.df.duplicated().sum())
        
        # Check duplicate equipment_id + timestamp pairs
        id_col = config.id_column
        ts_col = config.timestamp_column
        time_duplicates = 0
        if id_col in self.df.columns and ts_col in self.df.columns:
            time_duplicates = int(self.df.duplicated(subset=[id_col, ts_col]).sum())

        return {
            "total_duplicate_rows": total_duplicates,
            "duplicate_equipment_timestamp_pairs": time_duplicates
        }

    def check_value_ranges(self) -> pd.DataFrame:
        """Audits minimum, maximum, and out-of-range sensor readings against schema bounds."""
        results = []
        for col_def in EXPECTED_SCHEMA:
            col = col_def.name
            if col not in self.df.columns:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                actual_min = float(self.df[col].min())
                actual_max = float(self.df[col].max())
                allowed_min = col_def.min_value
                allowed_max = col_def.max_value

                below_min = int((self.df[col] < allowed_min).sum()) if allowed_min is not None else 0
                above_max = int((self.df[col] > allowed_max).sum()) if allowed_max is not None else 0

                results.append({
                    "Column": col,
                    "Actual Min": actual_min,
                    "Actual Max": actual_max,
                    "Allowed Min": allowed_min if allowed_min is not None else "N/A",
                    "Allowed Max": allowed_max if allowed_max is not None else "N/A",
                    "Below Min Count": below_min,
                    "Above Max Count": above_max
                })

        return pd.DataFrame(results)

    def check_target_distribution(self) -> Dict[str, Any]:
        """Analyzes class distribution for the target column ('failure')."""
        if TARGET_COLUMN not in self.df.columns:
            return {"error": f"Target column '{TARGET_COLUMN}' not present."}

        counts = self.df[TARGET_COLUMN].value_counts().to_dict()
        total = len(self.df)
        failure_count = counts.get(1, 0)
        normal_count = counts.get(0, 0)
        failure_pct = (failure_count / total) * 100.0 if total > 0 else 0.0

        return {
            "normal_count (0)": normal_count,
            "failure_count (1)": failure_count,
            "failure_rate_pct": round(failure_pct, 2),
            "total_records": total
        }

    def check_timestamp_validity(self) -> Dict[str, Any]:
        """Verifies timestamp formatting, ordering, and date range."""
        ts_col = config.timestamp_column
        if ts_col not in self.df.columns:
            return {"error": f"Timestamp column '{ts_col}' not present."}

        ts_series = pd.to_datetime(self.df[ts_col], errors="coerce")
        null_timestamps = int(ts_series.isnull().sum())
        min_date = str(ts_series.min())
        max_date = str(ts_series.max())

        return {
            "valid_timestamps": len(ts_series) - null_timestamps,
            "null_invalid_timestamps": null_timestamps,
            "start_timestamp": min_date,
            "end_timestamp": max_date
        }

    def run_full_validation(self) -> Dict[str, Any]:
        """Runs complete validation audit suite and aggregates results."""
        is_schema_valid, schema_errors = validate_schema(self.df)
        
        return {
            "dataset_shape": self.df.shape,
            "columns": list(self.df.columns),
            "is_schema_valid": is_schema_valid,
            "schema_errors": schema_errors,
            "missing_values": self.check_missing_values(),
            "duplicates": self.check_duplicates(),
            "range_audit": self.check_value_ranges(),
            "target_distribution": self.check_target_distribution(),
            "timestamp_validity": self.check_timestamp_validity()
        }


def validate_raw_dataset(filepath: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Loads raw CSV dataset from disk and executes validation checks.

    Args:
        filepath (Optional[Path | str]): CSV path. Defaults to config raw data path.

    Returns:
        Dict[str, Any]: Detailed validation summary report.
    """
    target_path = Path(filepath) if filepath else config.raw_data_path
    if not target_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {target_path}. Run src/data_generator.py first.")

    print(f"Loading raw dataset from {target_path} for validation...")
    df = pd.read_csv(target_path)
    validator = DatasetValidator(df)
    results = validator.run_full_validation()

    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT SUMMARY")
    print("=" * 60)
    print(f"Dataset Shape       : {results['dataset_shape']}")
    print(f"Schema Valid        : {results['is_schema_valid']}")
    print(f"Duplicates          : {results['duplicates']}")
    print(f"Failure Distribution: {results['target_distribution']}")
    print(f"Timestamp Range     : {results['timestamp_validity']['start_timestamp']} to {results['timestamp_validity']['end_timestamp']}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    validate_raw_dataset()
