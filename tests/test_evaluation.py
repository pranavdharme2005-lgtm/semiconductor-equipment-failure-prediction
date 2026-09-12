"""
Unit Tests for Model Evaluation, Reliability Analysis, and Threshold Optimization.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation import ModelEvaluator


def test_calculate_metrics():
    """Verifies precision, recall, F1, and AUC calculations."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.8, 0.9])

    metrics = ModelEvaluator.calculate_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 0.6666666666666666
    assert metrics["recall"] == 1.0
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics


def test_threshold_sweep_cost_minimization():
    """Verifies that evaluate_threshold_sweep generates valid threshold trade-offs."""
    y_true = np.array([0]*90 + [1]*10)
    y_prob = np.array([0.05]*85 + [0.3]*5 + [0.4]*2 + [0.85]*8)

    evaluator = ModelEvaluator()
    df_sweep = evaluator.evaluate_threshold_sweep(y_true, y_prob, thresholds=[0.2, 0.5, 0.8])

    assert len(df_sweep) == 3
    assert "total_financial_cost_usd" in df_sweep.columns
    assert "recall" in df_sweep.columns
    assert "precision" in df_sweep.columns
