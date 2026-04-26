"""
FastAPI application for Heart Disease Risk Screening
Production-grade REST API with comprehensive validation, error handling, and monitoring
"""

import time
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from src.api.config import settings, RISK_LABELS, RISK_DESCRIPTIONS
from src.api.schemas import (
    PatientRecord, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthCheckResponse, ErrorResponse, 
    DeferredPredictionResponse
)
from src.api.model_service import get_model_service
from src.api.logger import logger, log_prediction, log_error, log_validation_error

# Initialize FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Track API startup time
app_start_time = datetime.utcnow()


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    error_id = str(uuid.uuid4())
    log_error("VALIDATION_ERROR", str(exc), {"error_id": error_id})
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": "VALIDATION_ERROR",
            "error_message": str(exc),
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Handle runtime errors (model issues, etc.)"""
    error_id = str(uuid.uuid4())
    log_error("RUNTIME_ERROR", str(exc), {"error_id": error_id})
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "RUNTIME_ERROR",
            "error_message": str(exc),
            "error_id": error_id,
            "details": "Model prediction or processing failed",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors"""
    error_id = str(uuid.uuid4())
    log_error("INTERNAL_SERVER_ERROR", str(exc), {"error_id": error_id})
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "error_message": "An unexpected error occurred",
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get(
    "/api/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    tags=["System"],
    responses={
        200: {"description": "API is operational"},
        503: {"description": "API is degraded or offline"}
    }
)
async def health_check():
    """
    Check API health and model availability
    
    Returns operational status, model readiness, and uptime
    """
    try:
        model_service = get_model_service()
        health_status = model_service.health_check()
        
        uptime = (datetime.utcnow() - app_start_time).total_seconds()
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        
        return HealthCheckResponse(
            status=health_status["status"],
            version=settings.API_VERSION,
            model_loaded=model_service.model_loaded,
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@app.post(
    "/api/predict",
    response_model=PredictionResponse | DeferredPredictionResponse,
    summary="Single Patient Prediction",
    tags=["Prediction"],
    responses={
        200: {"description": "Prediction generated successfully"},
        422: {"description": "Invalid input data"},
        500: {"description": "Model prediction failed"}
    }
)
async def predict_single(patient: PatientRecord):
    """
    Predict heart disease risk for a single patient
    
    Accepts 13 clinical features and returns:
    - Risk classification (0=Low, 1=High)
    - Confidence score
    - Acceptance decision (Accepted if confidence >= threshold, else Deferred)
    
    **Confidence Threshold:** {settings.CONFIDENCE_THRESHOLD}
    
    Predictions below threshold are deferred for clinical review.
    """
    start_time = time.time()
    
    try:
        # Extract features in correct order
        features = [getattr(patient, field) for field in settings.FEATURE_ORDER]
        
        # Get model service
        model_service = get_model_service()
        
        # Make prediction
        prediction, confidence = model_service.predict_single(features)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Determine acceptance
        is_accepted = confidence >= settings.CONFIDENCE_THRESHOLD
        decision = "Accepted" if is_accepted else "Deferred"
        
        # Log prediction
        log_prediction(
            patient_id=str(uuid.uuid4()),
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            accepted=is_accepted
        )
        
        # Return response
        if is_accepted:
            return PredictionResponse(
                risk_class=prediction,
                risk_label=RISK_LABELS[prediction],
                risk_description=RISK_DESCRIPTIONS[prediction],
                confidence=confidence,
                decision=decision,
                processing_time_ms=processing_time_ms
            )
        else:
            return DeferredPredictionResponse(
                risk_class=None,
                risk_label=None,
                risk_description="Insufficient confidence for automated decision",
                confidence=confidence,
                decision=decision,
                clinical_note=f"Clinical review recommended (confidence: {confidence:.2%})",
                processing_time_ms=processing_time_ms
            )
    
    except ValueError as e:
        log_validation_error("patient_data", "validation", str(e))
        raise HTTPException(status_code=422, detail=str(e))
    
    except RuntimeError as e:
        log_error("PREDICTION_ERROR", str(e))
        raise HTTPException(status_code=500, detail="Model prediction failed")


@app.post(
    "/api/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Patient Predictions",
    tags=["Prediction"],
    responses={
        200: {"description": "Batch predictions completed"},
        422: {"description": "Invalid batch data"},
        500: {"description": "Batch processing failed"}
    }
)
async def predict_batch(batch_request: BatchPredictionRequest):
    """
    Predict heart disease risk for multiple patients (max 100 per request)
    
    Processes batch efficiently and returns individual predictions with batch statistics
    
    **Batch Size Limit:** 100 patients per request
    """
    start_time = time.time()
    batch_id = str(uuid.uuid4())
    
    try:
        # Extract features for all patients
        features_list = [
            [getattr(patient, field) for field in settings.FEATURE_ORDER]
            for patient in batch_request.patients
        ]
        
        # Get model service
        model_service = get_model_service()
        
        # Make batch prediction
        predictions, confidences = model_service.predict_batch(features_list)
        
        # Generate responses
        responses = []
        accepted_count = 0
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            is_accepted = conf >= settings.CONFIDENCE_THRESHOLD
            
            if is_accepted:
                accepted_count += 1
                response = PredictionResponse(
                    risk_class=pred,
                    risk_label=RISK_LABELS[pred],
                    risk_description=RISK_DESCRIPTIONS[pred],
                    confidence=conf,
                    decision="Accepted",
                    processing_time_ms=0  # Will update below
                )
            else:
                response = DeferredPredictionResponse(
                    risk_class=None,
                    risk_label=None,
                    risk_description="Insufficient confidence for automated decision",
                    confidence=conf,
                    decision="Deferred",
                    clinical_note=f"Clinical review recommended (confidence: {conf:.2%})",
                    processing_time_ms=0  # Will update below
                )
            
            responses.append(response)
        
        # Calculate timings
        batch_processing_time_ms = (time.time() - start_time) * 1000
        average_latency_ms = batch_processing_time_ms / len(batch_request.patients)
        
        # Update processing times
        for response in responses:
            response.processing_time_ms = average_latency_ms
        
        logger.info(
            f"Batch prediction completed: {batch_id}",
            extra={
                "batch_id": batch_id,
                "total_patients": len(batch_request.patients),
                "accepted": accepted_count,
                "deferred": len(batch_request.patients) - accepted_count,
                "total_time_ms": batch_processing_time_ms
            }
        )
        
        return BatchPredictionResponse(
            batch_id=batch_id,
            total_predictions=len(batch_request.patients),
            accepted_count=accepted_count,
            deferred_count=len(batch_request.patients) - accepted_count,
            predictions=responses,
            batch_processing_time_ms=batch_processing_time_ms,
            average_latency_ms=average_latency_ms
        )
    
    except ValueError as e:
        log_error("BATCH_VALIDATION_ERROR", str(e))
        raise HTTPException(status_code=422, detail=str(e))
    
    except RuntimeError as e:
        log_error("BATCH_PREDICTION_ERROR", str(e))
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get(
    "/api/model/info",
    summary="Model Information",
    tags=["Model"],
    responses={200: {"description": "Model metadata and feature importance"}}
)
async def model_info():
    """
    Get trained model information including:
    - Model type and configuration
    - Feature importance ranking
    - Training metadata
    """
    try:
        model_service = get_model_service()
        
        feature_importance = model_service.get_feature_importance()
        
        return {
            "model_info": {
                "model_type": "RandomForest (from Assignment 2)",
                "config": {
                    "n_estimators": 100,
                    "max_depth": 5
                },
                "dataset": "UCI Heart Disease (Cleveland)",
                "training_samples": 242,
                "validation_samples": 61
            },
            "feature_importance": feature_importance,
            "api_config": {
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "feature_count": len(settings.FEATURE_ORDER),
                "features": settings.FEATURE_ORDER
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api",
    summary="API Root",
    tags=["System"],
    responses={200: {"description": "API documentation links"}}
)
async def api_root():
    """API root endpoint with documentation links"""
    return {
        "message": "Heart Disease Risk Screening API v1.0",
        "documentation": "/api/docs",
        "endpoints": {
            "health": "/api/health",
            "predict_single": "/api/predict",
            "predict_batch": "/api/predict/batch",
            "model_info": "/api/model/info"
        }
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    try:
        logger.info("🚀 API Starting up...")
        model_service = get_model_service()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 API Shutting down...")


# ============================================================================
# CUSTOM OPENAPI SCHEMA
# ============================================================================

def custom_openapi():
    """Customize OpenAPI schema with additional metadata"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        description=settings.API_DESCRIPTION,
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
