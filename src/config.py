"""
Configuration Loader and Project Path Management Module.

Provides centralized configuration loading, dataclasses for structured parameters,
and automatic resolution of project paths.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml

# Resolve Project Root Directory (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigLoader:
    """Class to load and provide access to project configuration settings."""

    def __init__(self, config_path: Path | str | None = None):
        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file or returns default dictionary."""
        if not self.config_path.exists():
            print(f"Warning: Configuration file not found at {self.config_path}. Using fallback defaults.")
            return self._get_fallback_defaults()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading config file {self.config_path}: {e}. Using fallback defaults.")
            return self._get_fallback_defaults()

    def _get_fallback_defaults(self) -> Dict[str, Any]:
        """Fallback default configuration dictionary."""
        return {
            "project": {"name": "Semiconductor Equipment Failure Prediction", "version": "1.0.0"},
            "paths": {
                "raw_data": "data/raw/semiconductor_equipment_data.csv",
                "processed_data": "data/processed/semiconductor_equipment_cleaned.csv",
                "model_dir": "models/",
                "latest_model_path": "models/equipment_failure_model_latest.joblib",
            },
            "schema": {
                "id_column": "equipment_id",
                "timestamp_column": "timestamp",
                "target_column": "failure",
                "sensor_columns": [
                    "temperature", "pressure", "vibration", "voltage",
                    "current", "rpm", "flow_rate"
                ],
                "operational_columns": ["runtime_hours", "maintenance_count"],
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "primary_metric": "roc_auc",
            },
            "risk_scoring": {
                "thresholds": {"low": 0.30, "medium": 0.70}
            }
        }

    @property
    def raw_data_path(self) -> Path:
        return PROJECT_ROOT / self._config.get("paths", {}).get("raw_data", "data/raw/semiconductor_equipment_data.csv")

    @property
    def processed_data_path(self) -> Path:
        return PROJECT_ROOT / self._config.get("paths", {}).get("processed_data", "data/processed/semiconductor_equipment_cleaned.csv")

    @property
    def model_dir(self) -> Path:
        path = PROJECT_ROOT / self._config.get("paths", {}).get("model_dir", "models/")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def latest_model_path(self) -> Path:
        return PROJECT_ROOT / self._config.get("paths", {}).get("latest_model_path", "models/equipment_failure_model_latest.joblib")

    @property
    def id_column(self) -> str:
        return self._config.get("schema", {}).get("id_column", "equipment_id")

    @property
    def timestamp_column(self) -> str:
        return self._config.get("schema", {}).get("timestamp_column", "timestamp")

    @property
    def target_column(self) -> str:
        return self._config.get("schema", {}).get("target_column", "failure")

    @property
    def sensor_columns(self) -> List[str]:
        return self._config.get("schema", {}).get("sensor_columns", [
            "temperature", "pressure", "vibration", "voltage", "current", "rpm", "flow_rate"
        ])

    @property
    def operational_columns(self) -> List[str]:
        return self._config.get("schema", {}).get("operational_columns", ["runtime_hours", "maintenance_count"])

    @property
    def feature_columns(self) -> List[str]:
        return self.sensor_columns + self.operational_columns

    @property
    def validation_ranges(self) -> Dict[str, List[float]]:
        return self._config.get("data_validation", {}).get("ranges", {})

    @property
    def risk_thresholds(self) -> Dict[str, float]:
        return self._config.get("risk_scoring", {}).get("thresholds", {"low": 0.30, "medium": 0.70})


# Singleton Config Instance for easy import
config = ConfigLoader()
