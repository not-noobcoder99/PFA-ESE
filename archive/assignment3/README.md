# Assignment 3 – Model as a Service

**Course:** DS201 – Programming for AI  
**Assignment:** 3 – Heart Disease Risk Screening API  
**Deadline:** Mar 31, 2026  
**Rubric Weight:** 100% (25% API functionality, 20% robustness, 15% performance, 25% system reasoning, 15% reflection)

---

## Overview

Assignment 3 transforms the machine learning pipeline from Assignments 1-2 into a **production-grade REST API service**. The RandomForest model (from Assignment 2, AUC=0.9113) is deployed as a FastAPI microservice with:

- **REST API** with comprehensive input validation
- **Error handling** for malformed, extreme, and unexpected inputs
- **Stress testing** suite measuring latency, throughput, and robustness
- **OpenAPI/Swagger documentation** for client integration
- **Structured logging** for production monitoring
- **Deployment-ready architecture** for containerization

---

## Directory Structure

```
assignment3/
├── src/
│   ├── app.py                  # FastAPI application with all endpoints
│   ├── config.py               # Centralized settings and bounds
│   ├── schemas.py              # Pydantic validation models
│   ├── model_service.py        # ML model inference layer
│   ├── logger.py               # Structured JSON logging
│   └── __init__.py
│
├── tests/
│   ├── test_api.py             # pytest unit tests
│   ├── stress_test.py          # Latency, throughput, robustness tests
│   └── __init__.py
│
├── outputs/
│   ├── RandomForest.joblib     # Trained model (from Assignment 2)
│   ├── api.log                 # JSON structured logs
│   ├── stress_test_results.json# Performance metrics
│   └── evaluation_report.txt   # Report summary
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── DEPLOYMENT.md               # Docker & MLOps guidance

```

---

## Installation & Setup

### 1. Install Dependencies

```bash
cd "d:\ESE\PFA ESE\assignment3"
python -m pip install -r requirements.txt
```

### 2. Verify Model

Ensure `RandomForest.joblib` is present in `outputs/`:

```bash
ls outputs/RandomForest.joblib
```

### 3. Run API Server

```bash
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at: `http://localhost:8000/api/docs`

---

## API Endpoints

### Health Check
- **GET** `/api/health`
- Check API and model status
- **Response (200):** `{status, model_loaded, uptime_seconds}`

### Single Prediction
- **POST** `/api/predict`
- Predict heart disease risk for one patient
- **Input:** 13 clinical features (validated)
- **Response (200):** Prediction with confidence and decision

### Batch Prediction
- **POST** `/api/predict/batch`
- Predict for up to 100 patients
- **Input:** `{patients: [...]}`
- **Response (200):** Batch results with statistics

### Model Information
- **GET** `/api/model/info`
- Retrieve model metadata and feature importance
- **Response (200):** Model type, config, feature rankings

---

## Input Validation & Error Handling

### Comprehensive Validation

All inputs are validated using Pydantic schemas:

| Feature | Type | Valid Range |
|---------|------|-------------|
| age | int | 18–150 |
| sex | int | 0–1 (Female/Male) |
| cp | int | 0–3 (Chest pain type) |
| trestbps | int | 80–200 mmHg |
| chol | int | 100–600 mg/dl |
| fbs | int | 0–1 (Fasting BS >120) |
| restecg | int | 0–2 (ECG type) |
| thalach | int | 50–220 bpm |
| exang | int | 0–1 (Exercise angina) |
| oldpeak | float | 0.0–10.0 mm |
| slope | int | 0–2 (ST segment) |
| ca | int | 0–4 (Major vessels) |
| thal | int | 0–3 (Thalassemia) |

### Error Response Format

```json
{
  "error_code": "VALIDATION_ERROR",
  "error_message": "Invalid input: age must be positive",
  "error_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-03-30T14:30:45.123456Z"
}
```

### Exception Handling Strategy

1. **Validation Errors (422):** Pydantic schema violations
2. **Model Errors (500):** Inference or loading failures
3. **Generic Errors (500):** Unexpected server issues
4. All errors include unique error_id for debugging and logging

---

## Performance & Stress Testing

### Running Stress Tests

```bash
# Start API first
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000

# In another terminal
cd tests
python stress_test.py http://localhost:8000
```

### Test Suite Components

1. **Single Prediction Latency** (100 requests)
   - Measures: mean, median, min, max, p95, p99 latency
   - Typical: 10–50ms per prediction

2. **Batch Throughput**
   - Tests: batch sizes 10, 25, 50, 100
   - Measures: predictions/second, per-batch latency

3. **Error Handling Robustness**
   - Invalid age, sex, thresholds
   - Missing fields, out-of-range values
   - Validates 422 responses

4. **Malformed Input Handling**
   - Empty JSON, string data, empty batches, oversized batches
   - Verifies proper rejection

5. **Extreme Value Handling**
   - Very young (18), very old (120), high cholesterol
   - Verifies predictions still work at extremes

### Results Saved To

`outputs/stress_test_results.json` – Contains full timing and error statistics

---

## Unit Tests

### Running Tests

```bash
cd tests
pytest test_api.py -v
```

### Test Coverage

- **Health check** validation
- **Single predictions** with valid/invalid inputs
- **Batch predictions** up to max size (100)
- **Validation errors** for each field constraint
- **Response consistency** for repeated inputs
- **Model info** endpoint
- **Edge cases** (extreme ages, cholesterol, etc.)

---

## Logging & Monitoring

### Structured JSON Logging

All events logged to `outputs/api.log` in JSON format for parsing:

```json
{
  "timestamp": "2026-03-30T14:35:22.456Z",
  "level": "INFO",
  "name": "heart_disease_api",
  "message": "Prediction made",
  "patient_id": "abc-123",
  "prediction": 1,
  "confidence": 0.92,
  "processing_time_ms": 12.5
}
```

### Console Output

Human-readable logs to console during development.

---

## Confidence-Aware Prediction Logic

Model predictions include a **confidence score** based on max probability:

- **If confidence ≥ 0.7:** Prediction ACCEPTED → return risk class
- **If confidence < 0.7:** Prediction DEFERRED → flag for clinical review

Response includes clinical note recommending further evaluation.

---

## API Documentation

### Interactive Swagger UI

Visit: `http://localhost:8000/api/docs`

- Try all endpoints directly
- View request/response schemas
- Test with example data

### ReDoc Documentation

Visit: `http://localhost:8000/api/redoc`

- Alternative documentation format

---

## Deployment Considerations

### Requirements for Production

1. **Environment Variables** (.env)
   ```
   MODEL_PATH=/path/to/RandomForest.joblib
   CONFIDENCE_THRESHOLD=0.7
   LOG_LEVEL=INFO
   ```

2. **Production Server**
   ```bash
   gunicorn src.app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
   ```

3. **Docker** (see DEPLOYMENT.md)
   ```bash
   docker build -t heart-disease-api .
   docker run -p 8000:8000 heart-disease-api
   ```

4. **Monitoring**
   - Track `/api/health` status
   - Monitor `outputs/api.log` for errors
   - Set alerts on error rates

---

## Known Limitations & Future Work

### Current Limitations

1. **Single model serving** – Only RandomForest deployed (not ensemble)
2. **No caching** – Each request runs full inference
3. **Synchronous only** – No async batch processing
4. **Local model storage** – No S3/cloud model registry
5. **Basic auth** – No authentication/authorization

### Future Improvements

1. Model versioning & A/B testing
2. Asynchronous batch processing with job queue
3. Model monitoring & data drift detection
4. Cache predictions for identical inputs
5. Authentication via JWT tokens
6. Rate limiting per client
7. Model retraining pipeline trigger

---

## Summary of Deliverables

✅ **Code:**
- FastAPI application with 4 main endpoints
- Input validation schemas (Pydantic)
- Exception handling with structured errors
- Model service with batch inference
- Stress testing suite (latency, throughput, robustness)
- Unit tests (pytest)
- Structured JSON logging

✅ **Report (4-5 pages):**
- Architecture & API Design justification
- Notebook-to-Service Gap analysis
- Robustness & Reliability analysis
- Performance metrics & bottleneck identification
- Architectural reflection on decisions

---

## References

- **FastAPI:** https://fastapi.tiangolo.com/
- **Pydantic:** https://docs.pydantic.dev/
- **Uvicorn:** https://www.uvicorn.org/
- **Joblib:** https://joblib.readthedocs.io/

---

**Author:** DS201 Student  
**Last Updated:** 2026-03-30  
**Status:** Ready for Submission
