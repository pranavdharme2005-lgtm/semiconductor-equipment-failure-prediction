"""
Model Evaluation, Threshold Optimization, and Reliability Analysis Module.

Provides comprehensive classification metrics, ROC and Precision-Recall curve generation,
cost-sensitive threshold optimization, confusion matrix plotting, and threshold persistence.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix
)
import joblib

from src.config import config, PROJECT_ROOT


class ModelEvaluator:
    """
    Production Evaluation and Reliability Harness for Semiconductor Equipment Failure Prediction models.
    Supports detailed classification metrics, ROC/PR curves, and cost-sensitive threshold optimization.
    """

    # Estimated operational costs in USD
    COST_FALSE_NEGATIVE = 100000.0  # $100,000 per missed failure (broken wafers + tool damage)
    COST_FALSE_POSITIVE = 500.0     # $500 per false alarm (15-min technician sensor check)

    @staticmethod
    def calculate_metrics(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        y_prob: Optional[pd.Series | np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculates accuracy, precision, recall, F1, ROC-AUC, and PR-AUC."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        if y_prob is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
                metrics["pr_auc"] = float(auc(recall_arr, precision_arr))
            except Exception:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        return metrics

    @staticmethod
    def generate_confusion_matrix_dict(
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray
    ) -> Dict[str, int]:
        """Computes binary confusion matrix breakdown into TN, FP, FN, TP."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }

    def evaluate_threshold_sweep(
        self,
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Evaluates model performance across a spectrum of decision thresholds
        and calculates total estimated financial cost based on False Negatives and False Positives.

        Args:
            y_true: Ground truth binary labels.
            y_prob: Model predicted probabilities.
            thresholds: List of thresholds to evaluate.

        Returns:
            pd.DataFrame: Table comparing threshold metrics and financial cost.
        """
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

        records = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            cm = self.generate_confusion_matrix_dict(y_true, y_pred)
            tp, fp, fn, tn = cm["true_positives"], cm["false_positives"], cm["false_negatives"], cm["true_negatives"]

            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)

            # Financial cost calculation
            total_cost = (fn * self.COST_FALSE_NEGATIVE) + (fp * self.COST_FALSE_POSITIVE)

            records.append({
                "threshold": t,
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_financial_cost_usd": float(total_cost)
            })

        return pd.DataFrame(records)

    def plot_roc_and_pr_curves(
        self,
        models_dict: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        figures_dir: Path
    ) -> None:
        """Generates and saves ROC and Precision-Recall curve plots."""
        figures_dir.mkdir(parents=True, exist_ok=True)
        colors = {"Logistic Regression (Baseline)": "#0284c7", "Random Forest": "#e11d48", "Gradient Boosting": "#10b981"}

        # 1. ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            color = colors.get(name, "#64748b")
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linewidth=2.0, color=color)

        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1.0)
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(figures_dir / "08_roc_curve.png", dpi=150)
        plt.close()

        # 2. Precision-Recall Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)

            prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(rec_arr, prec_arr)
            color = colors.get(name, "#64748b")
            ax.plot(rec_arr, prec_arr, label=f"{name} (PR-AUC = {pr_auc:.4f})", linewidth=2.0, color=color)

        ax.set_title("Precision-Recall (PR) Curves (Imbalanced Failure Class Focus)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        plt.tight_layout()
        fig.savefig(figures_dir / "07_precision_recall_curve.png", dpi=150)
        plt.close()

    def plot_confusion_matrices_grid(
        self,
        models_dict: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        figures_dir: Path,
        operating_threshold: float = 0.40
    ) -> None:
        """Generates side-by-side confusion matrix heatmap visualizations."""
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4.5))
        if len(models_dict) == 1:
            axes = [axes]

        for idx, (name, model) in enumerate(models_dict.items()):
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)

            y_pred = (y_prob >= operating_threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                        xticklabels=["Normal (0)", "Failure (1)"], yticklabels=["Normal (0)", "Failure (1)"])
            axes[idx].set_title(f"{name}\n(Threshold = {operating_threshold:.2f})", fontweight="bold", fontsize=10)
            axes[idx].set_ylabel("Actual Label")
            axes[idx].set_xlabel("Predicted Label")

        plt.tight_layout()
        fig.savefig(figures_dir / "09_confusion_matrices.png", dpi=150)
        plt.close()

    def plot_threshold_tradeoff(self, df_sweep: pd.DataFrame, figures_dir: Path) -> None:
        """Plots Precision vs Recall vs Total Cost across threshold spectrum."""
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(df_sweep["threshold"], df_sweep["recall"], color="#0284c7", linewidth=2.0, marker="o", label="Recall (Failure Detection)")
        ax1.plot(df_sweep["threshold"], df_sweep["precision"], color="#e11d48", linewidth=2.0, marker="s", label="Precision (Alarm Accuracy)")
        ax1.set_xlabel("Decision Threshold", fontweight="bold")
        ax1.set_ylabel("Metric Score (0.0 - 1.0)", fontweight="bold")
        ax1.set_ylim(0.0, 1.05)

        ax2 = ax1.twinx()
        ax2.plot(df_sweep["threshold"], df_sweep["total_financial_cost_usd"] / 1000.0, color="#d97706", linewidth=2.0, linestyle="--", label="Total Financial Cost ($k)")
        ax2.set_ylabel("Estimated Financial Cost ($ Thousands USD)", color="#d97706", fontweight="bold")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        plt.title("Decision Threshold Trade-Off Analysis & Operational Cost Minimization", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(figures_dir / "10_threshold_cost_tradeoff.png", dpi=150)
        plt.close()


def run_evaluation_suite() -> Dict[str, Any]:
    """Factory execution function loading models, running threshold optimization, and saving figures."""
    from src.train import ModelTrainerPipeline

    print("Executing Stage 7 Model Evaluation & Reliability Suite...")
    trainer = ModelTrainerPipeline()
    df = trainer.load_dataset()
    _, _, test_df = trainer.time_aware_split(df)
    X_test, y_test = trainer.extract_features_and_target(test_df)

    # Load scaler and transform test set
    models_dir = config.model_dir
    scaler = joblib.load(models_dir / "scaler.joblib")
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Load trained model artifacts
    models_dict = {
        "Logistic Regression (Baseline)": joblib.load(models_dir / "logistic_regression_baseline.joblib"),
        "Random Forest": joblib.load(models_dir / "random_forest.joblib"),
        "Gradient Boosting": joblib.load(models_dir / "gradient_boosting.joblib")
    }

    evaluator = ModelEvaluator()
    figures_dir = PROJECT_ROOT / "notebooks" / "figures"

    # 1. Generate ROC & PR Curves
    print("Generating ROC and Precision-Recall curves...")
    evaluator.plot_roc_and_pr_curves(models_dict, X_test_scaled, y_test, figures_dir)

    # 2. Run Threshold Optimization on Champion Model (Logistic Regression)
    champ_model = models_dict["Logistic Regression (Baseline)"]
    champ_prob = champ_model.predict_proba(X_test_scaled)[:, 1]
    df_sweep = evaluator.evaluate_threshold_sweep(y_test, champ_prob)

    # Find cost-minimizing optimal threshold
    best_row = df_sweep.loc[df_sweep["total_financial_cost_usd"].idxmin()]
    optimal_threshold = float(best_row["threshold"])

    print(f"Optimal Decision Threshold Selected: {optimal_threshold:.2f}")
    print(f"  Recall at Threshold {optimal_threshold:.2f}    : {best_row['recall'] * 100:.2f}% (Missed Failures: {int(best_row['false_negatives'])})")
    print(f"  Precision at Threshold {optimal_threshold:.2f} : {best_row['precision'] * 100:.2f}% (False Alarms: {int(best_row['false_positives'])})")
    print(f"  Total Estimated Cost              : ${best_row['total_financial_cost_usd']:,.2f}")

    # 3. Generate Confusion Matrices & Threshold Trade-off Plots
    evaluator.plot_confusion_matrices_grid(models_dict, X_test_scaled, y_test, figures_dir, operating_threshold=optimal_threshold)
    evaluator.plot_threshold_tradeoff(df_sweep, figures_dir)

    # 4. Save Threshold Configuration JSON
    threshold_config = {
        "operating_threshold": optimal_threshold,
        "champion_model": "Logistic Regression (Baseline)",
        "metrics_at_operating_threshold": {
            "accuracy": float(best_row["accuracy"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1_score": float(best_row["f1_score"]),
            "true_positives": int(best_row["true_positives"]),
            "false_positives": int(best_row["false_positives"]),
            "false_negatives": int(best_row["false_negatives"]),
            "true_negatives": int(best_row["true_negatives"]),
            "total_cost_usd": float(best_row["total_financial_cost_usd"])
        },
        "risk_scoring_bounds": {
            "low_risk": [0.0, float(round(optimal_threshold * 0.75, 2))],
            "medium_risk": [float(round(optimal_threshold * 0.75, 2)), float(round(optimal_threshold, 2))],
            "high_risk": [float(round(optimal_threshold, 2)), 1.0]
        }
    }

    config_json_path = models_dir / "threshold_config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)

    print(f"Saved threshold configuration artifact to {config_json_path}")
    print("Stage 7 Model Evaluation & Reliability Analysis Complete.\n")

    return {
        "df_sweep": df_sweep,
        "optimal_threshold": optimal_threshold,
        "best_row": best_row,
        "config_json_path": str(config_json_path)
    }


if __name__ == "__main__":
    run_evaluation_suite()
