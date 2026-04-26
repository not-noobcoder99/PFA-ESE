"""
Logging configuration for API service
Structured logging with JSON output for production monitoring
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger
from src.config import settings


def setup_logging() -> logging.Logger:
    """Configure structured JSON logging for production"""
    
    # Ensure log directory exists
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("heart_disease_api")
    logger.setLevel(settings.LOG_LEVEL)
    
    # JSON formatter for structured logging
    json_formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    # File handler with JSON output
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with standard formatting
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


def log_prediction(
    patient_id: str,
    prediction: int,
    confidence: float,
    processing_time_ms: float,
    accepted: bool
):
    """Log prediction event with context"""
    logger.info(
        f"Prediction made",
        extra={
            "patient_id": patient_id,
            "prediction": prediction,
            "confidence": confidence,
            "processing_time_ms": processing_time_ms,
            "accepted": accepted,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_error(error_code: str, error_msg: str, details: dict = None):
    """Log error event with context"""
    logger.error(
        f"Error: {error_code}",
        extra={
            "error_code": error_code,
            "error_message": error_msg,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def log_validation_error(field: str, constraint: str, value: any):
    """Log validation error"""
    logger.warning(
        f"Validation error: {field}",
        extra={
            "error_type": "VALIDATION_ERROR",
            "field": field,
            "constraint": constraint,
            "value": str(value),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
