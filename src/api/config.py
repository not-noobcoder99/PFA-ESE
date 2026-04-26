"""
Configuration management for Heart Disease Risk Screening API
Centralized settings for models, validation, and service behavior
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # Service Configuration
    API_TITLE: str = "Heart Disease Risk Screening API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = (
        "Production-grade REST API for heart disease risk prediction "
        "with confidence scores and error handling"
    )
    
    # Model Configuration
    MODEL_PATH: str = str(PROJECT_ROOT / "models" / "RandomForest.joblib")
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_BATCH_SIZE: int = 100
    
    # Feature Configuration (13 features from UCI Heart Disease dataset)
    FEATURE_BOUNDS: dict = {
        "age": (29, 77),
        "sex": (0, 1),
        "cp": (0, 3),
        "trestbps": (90, 200),
        "chol": (126, 564),
        "fbs": (0, 1),
        "restecg": (0, 2),
        "thalach": (60, 202),
        "exang": (0, 1),
        "oldpeak": (0.0, 6.2),
        "slope": (0, 2),
        "ca": (0, 4),
        "thal": (0, 3),
    }
    
    FEATURE_ORDER: list = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(PROJECT_ROOT / "outputs" / "api" / "api.log")
    
    # API Configuration
    MAX_TIMEOUT: int = 30  # seconds
    ALLOWED_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Dictionary mapping class indices to risk labels
RISK_LABELS = {0: "Low Risk", 1: "High Risk"}
RISK_DESCRIPTIONS = {
    0: "Patient shows low risk indicators for heart disease",
    1: "Patient shows high risk indicators for heart disease"
}
