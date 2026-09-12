"""
Domain Feature Engineering Module for Semiconductor Equipment Telemetry.

Constructs rate-of-change lags, rolling moving averages, rolling volatility (std),
short vs long-term trends, nominal setpoint deviations, domain interaction ratios,
maintenance wear metrics, and expanding historical baselines while strictly avoiding future data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.config import config, PROJECT_ROOT


class EquipmentFeatureEngineer:
    """
    Feature Engineering Engine for Semiconductor Telemetry Data.
    Applies time-series lags, rolling windows, nominal setpoint deviations,
    and degradation ratios strictly backward-looking in time.
    """

    def __init__(self, short_window: int = 3, long_window: int = 6):
        self.short_window = short_window
        self.long_window = long_window

    def create_lags_and_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates 1-step lag and rate-of-change (difference) features per equipment_id.

        Features created:
        - temp_change_1h: Temperature difference from previous hour.
        - press_change_1h: Pressure difference from previous hour.
        - vib_change_1h: Vibration difference from previous hour.
        - current_change_1h: Current draw difference from previous hour.
        """
        df_feat = df.copy()
        id_col = config.id_column

        # Grouped 1-hour lags
        grouped = df_feat.groupby(id_col)
        
        df_feat["temp_lag1"] = grouped["temperature"].shift(1)
        df_feat["press_lag1"] = grouped["pressure"].shift(1)
        df_feat["vib_lag1"] = grouped["vibration"].shift(1)
        df_feat["current_lag1"] = grouped["current"].shift(1)

        # Differences (Rate of Change)
        df_feat["temp_change_1h"] = (df_feat["temperature"] - df_feat["temp_lag1"]).fillna(0.0)
        df_feat["press_change_1h"] = (df_feat["pressure"] - df_feat["press_lag1"]).fillna(0.0)
        df_feat["vib_change_1h"] = (df_feat["vibration"] - df_feat["vib_lag1"]).fillna(0.0)
        df_feat["current_change_1h"] = (df_feat["current"] - df_feat["current_lag1"]).fillna(0.0)

        # Drop intermediate raw lag columns to maintain clean feature set
        df_feat = df_feat.drop(columns=["temp_lag1", "press_lag1", "vib_lag1", "current_lag1"])

        return df_feat

    def create_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates backward-looking rolling means and rolling standard deviations per equipment_id.

        Features created:
        - vib_roll_mean_3h, vib_roll_mean_6h
        - vib_roll_std_3h, vib_roll_std_6h
        - temp_roll_mean_3h, temp_roll_mean_6h
        - temp_roll_std_3h, temp_roll_std_6h
        """
        df_feat = df.copy()
        id_col = config.id_column

        for window in [self.short_window, self.long_window]:
            grouped_vib = df_feat.groupby(id_col)["vibration"]
            grouped_temp = df_feat.groupby(id_col)["temperature"]

            df_feat[f"vib_roll_mean_{window}h"] = grouped_vib.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_feat[f"vib_roll_std_{window}h"] = grouped_vib.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0.0))

            df_feat[f"temp_roll_mean_{window}h"] = grouped_temp.transform(lambda x: x.rolling(window, min_periods=1).mean())
            df_feat[f"temp_roll_std_{window}h"] = grouped_temp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0.0))

        return df_feat

    def create_sensor_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates sensor trend indicators by comparing short-term vs long-term moving averages.

        Features created:
        - temp_trend_3h_6h: Difference between 3h and 6h rolling mean temperature.
        - vib_trend_3h_6h: Difference between 3h and 6h rolling mean vibration.
        """
        df_feat = df.copy()

        if f"temp_roll_mean_{self.short_window}h" in df_feat.columns and f"temp_roll_mean_{self.long_window}h" in df_feat.columns:
            df_feat["temp_trend_3h_6h"] = df_feat[f"temp_roll_mean_{self.short_window}h"] - df_feat[f"temp_roll_mean_{self.long_window}h"]

        if f"vib_roll_mean_{self.short_window}h" in df_feat.columns and f"vib_roll_mean_{self.long_window}h" in df_feat.columns:
            df_feat["vib_trend_3h_6h"] = df_feat[f"vib_roll_mean_{self.short_window}h"] - df_feat[f"vib_roll_mean_{self.long_window}h"]

        return df_feat

    def create_nominal_deviations_and_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates sensor setpoint deviations and domain interaction ratios.

        Features created:
        - voltage_dev: Absolute deviation from nominal 220V power supply.
        - rpm_dev: Speed drop from nominal 3200 RPM due to friction drag.
        - flow_rate_dev: Coolant/gas flow drop from nominal 45 L/min.
        - power_watts: Electrical power consumption (voltage * current).
        - thermal_strain_index: Combined thermal and mechanical stress (temperature * vibration).
        """
        df_feat = df.copy()

        # Deviations from nominal engineering operating setpoints
        df_feat["voltage_dev"] = (df_feat["voltage"] - 220.0).abs()
        df_feat["rpm_dev"] = 3200.0 - df_feat["rpm"]
        df_feat["flow_rate_dev"] = 45.0 - df_feat["flow_rate"]

        # Physical interaction ratios
        df_feat["power_watts"] = df_feat["voltage"] * df_feat["current"]
        df_feat["thermal_strain_index"] = df_feat["temperature"] * df_feat["vibration"]

        return df_feat

    def create_wear_and_maintenance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates equipment degradation wear ratios and maintenance frequency metrics.

        Features created:
        - runtime_since_maint: Cumulative runtime hours per maintenance service event.
        - maintenance_freq: Maintenance events per 1,000 runtime hours.
        """
        df_feat = df.copy()

        df_feat["runtime_since_maint"] = df_feat["runtime_hours"] / (df_feat["maintenance_count"] + 1.0)
        df_feat["maintenance_freq"] = df_feat["maintenance_count"] / ((df_feat["runtime_hours"] / 1000.0) + 1e-5)

        return df_feat

    def create_equipment_expanding_baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates equipment-level historical baseline statistics using expanding windows
        to prevent future data leakage (only uses past records <= current timestamp).

        Features created:
        - vib_expanding_mean: Historical mean vibration of equipment up to current timestamp.
        - temp_expanding_mean: Historical mean temperature of equipment up to current timestamp.
        - vib_expanding_ratio: Current vibration relative to equipment's historical baseline.
        """
        df_feat = df.copy()
        id_col = config.id_column

        grouped_vib = df_feat.groupby(id_col)["vibration"]
        grouped_temp = df_feat.groupby(id_col)["temperature"]

        df_feat["vib_expanding_mean"] = grouped_vib.transform(lambda x: x.expanding(min_periods=1).mean())
        df_feat["temp_expanding_mean"] = grouped_temp.transform(lambda x: x.expanding(min_periods=1).mean())
        
        df_feat["vib_expanding_ratio"] = df_feat["vibration"] / (df_feat["vib_expanding_mean"] + 1e-5)

        return df_feat

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes full feature engineering pipeline in sequential order.

        Args:
            df (pd.DataFrame): Preprocessed telemetry DataFrame.

        Returns:
            pd.DataFrame: Feature-enriched DataFrame.
        """
        df_transformed = df.copy()

        # Ensure chronological order per equipment_id before applying time-series transformations
        id_col = config.id_column
        ts_col = config.timestamp_column
        if id_col in df_transformed.columns and ts_col in df_transformed.columns:
            df_transformed[ts_col] = pd.to_datetime(df_transformed[ts_col])
            df_transformed = df_transformed.sort_values(by=[id_col, ts_col]).reset_index(drop=True)

        df_transformed = self.create_lags_and_changes(df_transformed)
        df_transformed = self.create_rolling_statistics(df_transformed)
        df_transformed = self.create_sensor_trends(df_transformed)
        df_transformed = self.create_nominal_deviations_and_interactions(df_transformed)
        df_transformed = self.create_wear_and_maintenance_metrics(df_transformed)
        df_transformed = self.create_equipment_expanding_baselines(df_transformed)

        return df_transformed

    def process_and_save(
        self,
        input_path: Optional[Path | str] = None,
        output_path: Optional[Path | str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads preprocessed CSV, applies feature engineering, saves feature CSV dataset.

        Args:
            input_path: Source processed CSV path.
            output_path: Destination features CSV path.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Feature-enriched DataFrame, Summary stats)
        """
        src_path = Path(input_path) if input_path else config.processed_data_path
        if not src_path.exists():
            raise FileNotFoundError(f"Processed data file not found at {src_path}. Run Stage 3 preprocessing first.")

        print(f"Loading preprocessed dataset from {src_path}...")
        df_raw = pd.read_csv(src_path)
        original_cols = list(df_raw.columns)

        print("Executing feature engineering pipeline...")
        df_features = self.transform(df_raw)
        final_cols = list(df_features.columns)
        engineered_cols = [col for col in final_cols if col not in original_cols]

        dest_path = Path(output_path) if output_path else (PROJECT_ROOT / "data" / "processed" / "semiconductor_equipment_features.csv")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(dest_path, index=False)

        summary_stats = {
            "num_records": len(df_features),
            "num_original_features": len(original_cols),
            "num_engineered_features": len(engineered_cols),
            "total_features": len(final_cols),
            "original_features": original_cols,
            "engineered_features": engineered_cols,
            "output_path": str(dest_path)
        }

        return df_features, summary_stats


def run_feature_engineering_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Factory wrapper executing feature engineering pipeline."""
    fe = EquipmentFeatureEngineer()
    df_features, stats = fe.process_and_save()

    print("\n" + "=" * 60)
    print("STAGE 5: FEATURE ENGINEERING PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total Dataset Records     : {stats['num_records']:,}")
    print(f"Original Base Columns     : {stats['num_original_features']}")
    print(f"Engineered Features Added : {stats['num_engineered_features']}")
    print(f"Total Columns In Dataset  : {stats['total_features']}")
    print(f"Feature CSV Saved To      : {stats['output_path']}")
    print("=" * 60 + "\n")

    return df_features, stats


if __name__ == "__main__":
    run_feature_engineering_pipeline()
