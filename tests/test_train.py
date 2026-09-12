"""
Unit Tests for Machine Learning Training Pipeline & Model Persistence.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.train import ModelTrainerPipeline
from src.config import config


def test_time_aware_split():
    """Verifies that time_aware_split maintains chronological order and proportions."""
    pipeline = ModelTrainerPipeline()
    df = pipeline.load_dataset()
    
    train_df, val_df, test_df = pipeline.time_aware_split(df, train_ratio=0.70, val_ratio=0.15)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert len(train_df) == 7000
    assert len(val_df) == 1500
    assert len(test_df) == 1500

    # Ensure no temporal overlap
    for eq_id in df[config.id_column].unique():
        tr_max = train_df[train_df[config.id_column] == eq_id][config.timestamp_column].max()
        val_min = val_df[val_df[config.id_column] == eq_id][config.timestamp_column].min()
        test_min = test_df[test_df[config.id_column] == eq_id][config.timestamp_column].min()
        
        assert tr_max <= val_min
        assert val_min <= test_min


def test_model_training_pipeline_execution():
    """Verifies that training pipeline runs and saves model artifacts."""
    pipeline = ModelTrainerPipeline()
    results = pipeline.train_and_evaluate_all_models()

    assert "eval_summary" in results
    assert "champion_model_name" in results
    assert config.latest_model_path.exists()
