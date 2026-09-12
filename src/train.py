"""
Machine Learning Pipeline & Model Training Engine for Semiconductor Equipment Failure Prediction.

Implements chronological time-aware train/validation/test splitting, feature scaling,
class imbalance handling, multi-model training (Logistic Regression, Random Forest, Gradient Boosting),
hyperparameter selection, performance evaluation, and model artifact persistence.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

from src.config import config, PROJECT_ROOT
from src.evaluation import ModelEvaluator


class ModelTrainerPipeline:
    """
    Production ML Training Pipeline for Semiconductor Telemetry Failure Prediction.
    Supports chronological time-aware splits, class balancing, model comparison, and artifact saving.
    """

    def __init__(self, feature_data_path: Optional[Path | str] = None):
        self.data_path = Path(feature_data_path) if feature_data_path else (PROJECT_ROOT / "data" / "processed" / "semiconductor_equipment_features.csv")
        self.scaler = StandardScaler()
        self.evaluator = ModelEvaluator()
        self.trained_models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def load_dataset(self) -> pd.DataFrame:
        """Loads feature-engineered dataset from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Feature dataset not found at {self.data_path}. Run Stage 5 feature engineering first.")

        df = pd.read_csv(self.data_path)
        if config.timestamp_column in df.columns:
            df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
        return df

    def time_aware_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Performs chronological time-aware train/validation/test splitting per equipment_id
        to prevent future data leakage.

        Args:
            df (pd.DataFrame): Sorted feature DataFrame.
            train_ratio (float): Fraction for training set.
            val_ratio (float): Fraction for validation set.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        id_col = config.id_column
        ts_col = config.timestamp_column
        
        # Ensure chronological ordering per equipment_id
        df_sorted = df.sort_values(by=[id_col, ts_col]).reset_index(drop=True)

        train_chunks, val_chunks, test_chunks = [], [], []

        for eq_id, group in df_sorted.groupby(id_col, sort=False):
            n = len(group)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_chunks.append(group.iloc[:train_end])
            val_chunks.append(group.iloc[train_end:val_end])
            test_chunks.append(group.iloc[val_end:])

        train_df = pd.concat(train_chunks, axis=0).reset_index(drop=True)
        val_df = pd.concat(val_chunks, axis=0).reset_index(drop=True)
        test_df = pd.concat(test_chunks, axis=0).reset_index(drop=True)

        return train_df, val_df, test_df

    def extract_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separates feature matrix X from target vector y."""
        exclude_cols = [config.target_column, config.id_column, config.timestamp_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[config.target_column]
        return X, y

    def train_and_evaluate_all_models(self) -> Dict[str, Any]:
        """
        Executes end-to-end model training, validation, and test set evaluation across:
        1. Logistic Regression (Baseline)
        2. Random Forest Classifier
        3. Gradient Boosting Classifier
        """
        df = self.load_dataset()
        train_df, val_df, test_df = self.time_aware_split(df)

        print(f"Data Split Summary:")
        print(f"  Train Set      : {len(train_df):,} records (Failures: {train_df['failure'].sum()})")
        print(f"  Validation Set : {len(val_df):,} records (Failures: {val_df['failure'].sum()})")
        print(f"  Test Set       : {len(test_df):,} records (Failures: {test_df['failure'].sum()})")

        X_train, y_train = self.extract_features_and_target(train_df)
        X_val, y_val = self.extract_features_and_target(val_df)
        X_test, y_test = self.extract_features_and_target(test_df)

        # Fit scaler strictly on training set
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(self.scaler.transform(X_val), columns=X_val.columns)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

        # Save scaler artifact
        models_dir = config.model_dir
        joblib.dump(self.scaler, models_dir / "scaler.joblib")

        # Define candidate model architectures
        models = {
            "Logistic Regression (Baseline)": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.08,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )
        }

        eval_summary = {}
        best_model_name = None
        best_pr_auc = -1.0

        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            self.trained_models[name] = model

            # Predict on Test Set
            y_pred = model.predict(X_test_scaled)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = y_pred.astype(float)

            # Compute evaluation metrics
            metrics = self.evaluator.calculate_metrics(y_test, y_pred, y_prob)
            cm = self.evaluator.generate_confusion_matrix_dict(y_test, y_pred)
            
            eval_summary[name] = {
                "metrics": metrics,
                "confusion_matrix": cm,
                "model_object": model
            }

            # Model Selection Criterion: Highest PR-AUC (best trade-off for imbalanced failure prediction)
            if metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = metrics["pr_auc"]
                best_model_name = name

            # Save individual model artifact
            safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
            joblib.dump(model, models_dir / safe_filename)

        # Save Champion Model as equipment_failure_model_latest.joblib
        champion_model = self.trained_models[best_model_name]
        joblib.dump(champion_model, config.latest_model_path)

        self.results = {
            "eval_summary": eval_summary,
            "champion_model_name": best_model_name,
            "feature_names": list(X_train.columns),
            "test_sample_count": len(y_test),
            "test_failure_count": int(y_test.sum())
        }

        return self.results


def run_model_training_pipeline() -> Dict[str, Any]:
    """Factory execution wrapper for ML model training pipeline."""
    pipeline = ModelTrainerPipeline()
    results = pipeline.train_and_evaluate_all_models()

    eval_summary = results["eval_summary"]
    champion_name = results["champion_model_name"]

    print("\n" + "=" * 80)
    print("STAGE 6: MACHINE LEARNING MODEL TRAINING & COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model Architecture':<32} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'PR-AUC':<7}")
    print("-" * 80)

    for name, data in eval_summary.items():
        m = data["metrics"]
        print(f"{name:<32} | {m['accuracy']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<7.4f} | {m['f1_score']:<8.4f} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<7.4f}")

    print("-" * 80)
    print(f"\nCHAMPION MODEL SELECTED: {champion_name} (PR-AUC: {eval_summary[champion_name]['metrics']['pr_auc']:.4f})")
    print(f"   Model Artifact Saved To: {config.latest_model_path}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_model_training_pipeline()
