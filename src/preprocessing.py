"""
Data Preprocessing Pipeline for Semiconductor Equipment Telemetry.

Implements chronological sorting, deduplication, group-wise time-series missing value imputation,
physical domain bounds clipping (distinguishing data errors from true failure anomalies),
categorical encoding, and data-leakage-free preprocessing contracts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.config import config
from src.schema import validate_schema, EXPECTED_SCHEMA, FEATURE_COLUMNS, TARGET_COLUMN


class EquipmentDataPreprocessor:
    """
    Production Preprocessing Engine for Semiconductor Telemetry Datasets.
    Performs chronological sorting, deduplication, time-series imputation,
    outlier handling, and scaling without data leakage.
    """

    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = config.feature_columns
        self.is_fitted: bool = False
        self.stats: Dict[str, Any] = {}

    def load_and_validate(self, filepath: Optional[Path | str] = None) -> pd.DataFrame:
        """Loads raw CSV dataset and validates basic schema presence."""
        target_path = Path(filepath) if filepath else config.raw_data_path
        if not target_path.exists():
            raise FileNotFoundError(f"Raw dataset file not found at: {target_path}")

        df = pd.read_csv(target_path)
        is_valid, errors = validate_schema(df)
        if not is_valid:
            print(f"Warning: Raw dataset schema validation returned errors: {errors}")

        return df

    def parse_and_sort_chronologically(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses timestamp column into pandas Datetime and sorts chronologically per equipment_id.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Chronologically sorted DataFrame.
        """
        df_sorted = df.copy()
        ts_col = config.timestamp_column
        id_col = config.id_column

        if ts_col in df_sorted.columns:
            df_sorted[ts_col] = pd.to_datetime(df_sorted[ts_col])

        sort_cols = [col for col in [id_col, ts_col] if col in df_sorted.columns]
        if sort_cols:
            df_sorted = df_sorted.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

        return df_sorted

    def deduplicate_records(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Identifies and removes duplicate records based on equipment_id and timestamp.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            Tuple[pd.DataFrame, int]: (Deduplicated DataFrame, count of duplicates removed)
        """
        id_col = config.id_column
        ts_col = config.timestamp_column
        subset_cols = [col for col in [id_col, ts_col] if col in df.columns]

        if subset_cols:
            initial_count = len(df)
            df_dedup = df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)
            duplicates_removed = initial_count - len(df_dedup)
        else:
            initial_count = len(df)
            df_dedup = df.drop_duplicates(keep="first").reset_index(drop=True)
            duplicates_removed = initial_count - len(df_dedup)

        return df_dedup, duplicates_removed

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values using time-series forward fill (ffill) and backward fill (bfill)
        grouped by equipment_id, preventing cross-equipment data leakage. Median fallback applied if needed.

        Args:
            df (pd.DataFrame): DataFrame with potential NaNs.

        Returns:
            pd.DataFrame: Fully imputed DataFrame.
        """
        df_imputed = df.copy()
        id_col = config.id_column
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns

        # Grouped time-series ffill and bfill per equipment unit
        if id_col in df_imputed.columns:
            df_imputed[numeric_cols] = df_imputed.groupby(id_col)[numeric_cols].transform(
                lambda group: group.ffill().bfill()
            )

        # Fallback median imputation for any remaining NaNs
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)

        return df_imputed

    def handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Handles outliers by distinguishing physical data errors from genuine equipment failure anomalies:
        - Physical Data Errors (e.g. negative temperature, negative pressure, impossible voltage bounds):
          Clipped to allowable physical domain boundaries defined in schema.
        - Genuine Degradation Anomalies (e.g. high temperature or vibration preceding failure):
          Preserved intact as critical predictive signals.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: (Clipped DataFrame, dictionary of clipped count per feature)
        """
        df_cleaned = df.copy()
        clip_counts = {}

        for col_def in EXPECTED_SCHEMA:
            col = col_def.name
            if col not in df_cleaned.columns or not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue

            min_bound = col_def.min_value
            max_bound = col_def.max_value

            if min_bound is not None or max_bound is not None:
                initial_series = df_cleaned[col].copy()
                
                # Clip values to physical schema domain
                lower = min_bound if min_bound is not None else -np.inf
                upper = max_bound if max_bound is not None else np.inf
                
                df_cleaned[col] = df_cleaned[col].clip(lower=lower, upper=upper)
                
                clipped_num = int((initial_series != df_cleaned[col]).sum())
                if clipped_num > 0:
                    clip_counts[col] = clipped_num

        return df_cleaned, clip_counts

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical variables (e.g. equipment_id) into numeric code columns.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with equipment_code added.
        """
        df_encoded = df.copy()
        id_col = config.id_column

        if id_col in df_encoded.columns:
            if not self.is_fitted:
                df_encoded["equipment_code"] = self.label_encoder.fit_transform(df_encoded[id_col])
            else:
                # Handle potential unseen IDs during inference
                df_encoded["equipment_code"] = df_encoded[id_col].apply(
                    lambda x: self.label_encoder.transform([x])[0] if x in self.label_encoder.classes_ else -1
                )

        return df_encoded

    def process_pipeline(
        self,
        input_path: Optional[Path | str] = None,
        output_path: Optional[Path | str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes full preprocessing pipeline sequentially on raw dataset.

        Args:
            input_path: Source raw CSV path.
            output_path: Destination processed CSV path.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Processed DataFrame, Summary statistics dictionary)
        """
        raw_df = self.load_and_validate(input_path)
        rows_before = len(raw_df)
        missing_before = int(raw_df.isnull().sum().sum())

        # 1. Parse timestamps & sort chronologically per equipment_id
        sorted_df = self.parse_and_sort_chronologically(raw_df)

        # 2. Deduplicate records
        dedup_df, duplicates_removed = self.deduplicate_records(sorted_df)

        # 3. Impute missing values
        imputed_df = self.impute_missing_values(dedup_df)
        missing_after = int(imputed_df.isnull().sum().sum())

        # 4. Handle outliers & clip physical errors
        cleaned_df, clip_counts = self.handle_outliers(imputed_df)

        # 5. Encode categorical variables
        final_df = self.encode_categorical_features(cleaned_df)
        rows_after = len(final_df)

        # Save processed dataset
        target_output = Path(output_path) if output_path else config.processed_data_path
        target_output.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(target_output, index=False)

        summary_stats = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "duplicates_removed": duplicates_removed,
            "clip_counts": clip_counts,
            "final_columns": list(final_df.columns),
            "output_path": str(target_output)
        }

        self.stats = summary_stats
        return final_df, summary_stats


def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Factory execution wrapper running preprocessing pipeline."""
    preprocessor = EquipmentDataPreprocessor()
    df_processed, stats = preprocessor.process_pipeline()

    print("\n" + "=" * 60)
    print("STAGE 3: DATA PREPROCESSING PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Rows Before Preprocessing : {stats['rows_before']}")
    print(f"Rows After Preprocessing  : {stats['rows_after']}")
    print(f"Missing Values Before     : {stats['missing_before']}")
    print(f"Missing Values After      : {stats['missing_after']}")
    print(f"Duplicates Removed        : {stats['duplicates_removed']}")
    print(f"Outliers / Error Clipped  : {stats['clip_counts']}")
    print(f"Processed CSV Saved To    : {stats['output_path']}")
    print("=" * 60 + "\n")

    return df_processed, stats


if __name__ == "__main__":
    run_preprocessing_pipeline()
