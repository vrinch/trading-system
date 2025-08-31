"""
Configuration management for ML pipeline: storage, training, and model promotion.
"""

from enum import Enum
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, RedisDsn, PostgresDsn


class MLBackend(str, Enum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    TORCH = "torch"


class MLPipelineConfig(BaseSettings):
    service_name: str = "ml-pipeline"
    service_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")

    postgres_url: PostgresDsn = Field(
        default="postgresql://ml_user:ml_pass@localhost:5432/trading_db", env="POSTGRES_URL"
    )
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/13", env="REDIS_URL"
    )
    s3_endpoint: str = Field(default="http://localhost:9000", env="MINIO_ENDPOINT")
    s3_access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    s3_secret_key: str = Field(default="minioadmin123", env="MINIO_SECRET_KEY")
    mlflow_tracking_uri: str = Field(default="sqlite:///mlruns.db", env="MLFLOW_TRACKING_URI")

    # Features & Dataset
    feature_store_path: str = Field(default="./features/", env="FEATURE_STORE_PATH")
    model_artifact_path: str = Field(default="./model_artifacts/", env="MODEL_ARTIFACT_PATH")
    max_training_samples: int = Field(default=100000, env="MAX_TRAINING_SAMPLES")
    enable_model_drift_monitoring: bool = Field(default=True, env="ENABLE_MODEL_DRIFT_MONITOR")

    # Model selection/promote
    model_selection_metric: str = Field(default="sharpe_ratio", env="MODEL_SELECTION_METRIC")
    drift_detection_threshold: float = Field(default=0.05, env="DRIFT_DETECTION_THRESHOLD")

    # Scheduler
    auto_retrain_schedule: str = Field(default="0 3 * * *", env="RETRAIN_SCHEDULE")  # 3am daily

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_config() -> MLPipelineConfig:
    return MLPipelineConfig()