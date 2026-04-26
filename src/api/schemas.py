"""
Pydantic schemas for input validation and response serialization
Enforces strict type checking, bounds validation, and clear API contracts
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class Sex(int, Enum):
    """Biological sex encoding"""
    FEMALE = 0
    MALE = 1


class ChestPain(int, Enum):
    """Chest pain type"""
    TYPICAL_ANGINA = 0
    ATYPICAL_ANGINA = 1
    NOT_ANGINAL_PAIN = 2
    ASYMPTOMATIC = 3


class PatientRecord(BaseModel):
    """
    Single patient health record for heart disease risk prediction
    All fields must be within clinical bounds
    """
    age: int = Field(
        ..., gt=0, le=150,
        description="Patient age in years (1-150)"
    )
    sex: int = Field(
        ..., ge=0, le=1,
        description="Biological sex: 0=Female, 1=Male"
    )
    cp: int = Field(
        ..., ge=0, le=3,
        description="Chest pain type: 0=Typical, 1=Atypical, 2=Not Anginal, 3=Asymptomatic"
    )
    trestbps: int = Field(
        ..., ge=80, le=210,
        description="Resting blood pressure (mmHg)"
    )
    chol: int = Field(
        ..., ge=100, le=600,
        description="Serum cholesterol (mg/dl)"
    )
    fbs: int = Field(
        ..., ge=0, le=1,
        description="Fasting blood sugar > 120 mg/dl: 0=False, 1=True"
    )
    restecg: int = Field(
        ..., ge=0, le=2,
        description="Resting ECG: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy"
    )
    thalach: int = Field(
        ..., ge=50, le=220,
        description="Max heart rate achieved (bpm)"
    )
    exang: int = Field(
        ..., ge=0, le=1,
        description="Exercise-induced angina: 0=No, 1=Yes"
    )
    oldpeak: float = Field(
        ..., ge=0, le=10,
        description="ST depression induced by exercise (mm)"
    )
    slope: int = Field(
        ..., ge=0, le=2,
        description="Slope of ST segment: 0=Upsloping, 1=Flat, 2=Downsloping"
    )
    ca: int = Field(
        ..., ge=0, le=4,
        description="Number of major vessels (0-4) colored by fluoroscopy"
    )
    thal: int = Field(
        ..., ge=0, le=3,
        description="Thalassemia: 0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Unknown"
    )

    @validator("age")
    def validate_age(cls, v):
        if v < 18:
            raise ValueError("Age must be >= 18 (adult patient)")
        if v > 120:
            raise ValueError("Age exceeds reasonable bounds (> 120)")
        return v

    @validator("trestbps")
    def validate_bp(cls, v):
        if v < 80 or v > 200:
            raise ValueError("Blood pressure outside clinical range (80-200 mmHg)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "sex": 1,
                "cp": 0,
                "trestbps": 140,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 130,
                "exang": 0,
                "oldpeak": 2.6,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction result with confidence and metadata"""
    risk_class: int = Field(
        ..., ge=0, le=1,
        description="Predicted class: 0=Low Risk, 1=High Risk"
    )
    risk_label: str = Field(
        ..., description="Human-readable risk label"
    )
    risk_description: str = Field(
        ..., description="Clinical interpretation of prediction"
    )
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Model confidence in prediction (0-1)"
    )
    decision: str = Field(
        ..., enum=["Accepted", "Deferred"],
        description="Whether prediction is accepted or deferred for review"
    )
    processing_time_ms: float = Field(
        ..., description="API response time in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "risk_class": 1,
                "risk_label": "High Risk",
                "risk_description": "Patient shows high risk indicators for heart disease",
                "confidence": 0.92,
                "decision": "Accepted",
                "processing_time_ms": 12.5
            }
        }


class DeferredPredictionResponse(PredictionResponse):
    """Deferred prediction when confidence is below threshold"""
    risk_class: Optional[int] = None
    risk_label: Optional[str] = None
    clinical_note: str = Field(
        default="Further medical evaluation recommended",
        description="Recommendation for clinical follow-up"
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple patients"""
    patients: List[PatientRecord] = Field(
        ..., max_items=100,
        description="List of patient records (max 100 per request)"
    )

    @validator("patients")
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch must contain at least 1 patient")
        if len(v) > 100:
            raise ValueError("Batch size exceeds maximum (100 patients)")
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction results"""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_predictions: int = Field(..., description="Total predictions in batch")
    accepted_count: int = Field(..., description="Number of accepted predictions")
    deferred_count: int = Field(..., description="Number of deferred predictions")
    predictions: List[PredictionResponse] = Field(..., description="Individual predictions")
    batch_processing_time_ms: float = Field(..., description="Total batch processing time")
    average_latency_ms: float = Field(..., description="Average per-prediction latency")


class HealthCheckResponse(BaseModel):
    """Service health check response"""
    status: str = Field(..., enum=["healthy", "degraded", "offline"])
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    uptime_seconds: float = Field(..., description="API uptime in seconds")


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error_code: str = Field(..., description="Error code identifier")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error context")
    timestamp: str = Field(..., description="ISO 8601 timestamp of error")

    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "error_message": "Invalid input: age must be positive",
                "details": {"field": "age", "constraint": "gt:0"},
                "timestamp": "2026-03-30T14:30:45.123456Z"
            }
        }
