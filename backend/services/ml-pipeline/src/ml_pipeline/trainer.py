"""
Robust ML training pipeline: handles dataset loading, feature processing,
hyperparameter tuning, training, and artifact registration.
"""

import os
import time
from typing import Any, Dict, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import lightgbm as lgb
from .config import get_config
from .validator import validate_model

config = get_config()


def load_features(feature_path: str) -> pd.DataFrame:
    """Load prepared feature set from storage."""
    return pd.read_parquet(feature_path)


def split_dataset(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, seed: int = 42
) -> Any:
    """Split features into train/test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def train_lightgbm(
    X_train, y_train, X_val, y_val, params: Optional[Dict[str, Any]] = None
) -> lgb.Booster:
    """Train LightGBM model and return booster."""
    params = params or {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.01,
        "num_leaves": 64,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=1000,
        early_stopping_rounds=40,
        verbose_eval=100,
    )
    return model


def run_training(
    feature_file: str,
    target_col: str,
    model_type: str = "lightgbm",
    artifact_subpath: str = "",
    custom_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Main training routine with MLflow logging and artifact registration."""
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    with mlflow.start_run(run_name=f"{model_type}_{os.path.basename(feature_file)}"):
        features = load_features(feature_file)
        X_train, X_test, y_train, y_test = split_dataset(features, target_col)
        if model_type == "lightgbm":
            model = train_lightgbm(X_train, y_train, X_test, y_test, custom_params)
            y_pred = model.predict(X_test)
            score = np.corrcoef(y_pred, y_test)[0, 1]
            mlflow.log_metric("test_corr", score)
            model_path = f"{config.model_artifact_path}/{artifact_subpath}/lightgbm.pkl"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
        else:
            raise NotImplementedError("Only LightGBM supported in this example.")

        # Model validation/registration for deployment
        validate_model(model_path, model_type=model_type)

        return {
            "model_path": model_path,
            "metric": score,
            "run_id": mlflow.active_run().info.run_id,
        }