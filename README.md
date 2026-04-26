# Heart Disease Risk Screening - Semester Project

This repository is a unified, production-style version of all completed course work.

It combines:
- Phase 1: Baseline ML pipeline with confidence-aware decision deferral
- Phase 2: Multi-model comparison, evaluation, and model serialization
- Phase 3: FastAPI model-serving layer with validation, logging, and stress testing

## Why this structure

Instead of assignment-specific folders, all runnable deliverables are grouped by project concern:
- `src/phase1` for baseline training workflow
- `src/phase2` for comparative modeling workflow
- `src/api` for deployment/service workflow
- `data`, `models`, `outputs`, `tests`, and `docs` for clean engineering separation

Original assignment folders have been archived under `archive/` for traceability.

## Final Project Structure

```text
PFA-ESE/
  archive/
    assignment3/
    PFA ESE/
  data/
    heart.csv
  docs/
  models/
    RandomForest.joblib
  outputs/
    phase1/
    phase2/
    api/
  scripts/
    run_phase1.ps1
    run_phase2.ps1
    run_api.ps1
    run_all.ps1
  src/
    phase1/
    phase2/
    api/
  tests/
    phase2/
    api/
```

## Environment Setup

```powershell
cd "c:\Users\hp\OneDrive\Desktop\PFA-ESE"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Workflows

### Phase 1 (baseline)
```powershell
.\scripts\run_phase1.ps1
```

### Phase 2 (comparison)
```powershell
.\scripts\run_phase2.ps1
```

### API (Phase 3)
```powershell
.\scripts\run_api.ps1
```

### Full pipeline
```powershell
.\scripts\run_all.ps1
```

## Run Tests

```powershell
python -m pytest tests\phase2 -v
python -m pytest tests\api\test_api.py -v
```

## Docker

Build the API image:

```powershell
docker build -t heart-disease-semester-project .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 heart-disease-semester-project
```

Then open:
- Swagger UI: `http://localhost:8000/api/docs`
- Health check: `http://localhost:8000/api/health`

Docker Desktop or another working Docker Engine is required to run these commands locally.

## API Endpoints

- `GET /api/health`
- `POST /api/predict`
- `POST /api/predict/batch`
- `GET /api/model/info`

After API startup:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

## Semester Deliverables Covered

- Modular ML pipeline and leakage-safe preprocessing
- Comparative model evaluation (accuracy, F1, AUC, MCC, confusion matrix)
- Serialization and reproducibility checks
- Production-oriented REST API with robust validation and structured logging
- Unit and stress testing
- Deployment-ready Dockerfile and CI workflow

## Notes

- The API uses `models/RandomForest.joblib` produced during model comparison.
- Generated artifacts are written to `outputs/phase1`, `outputs/phase2`, and `outputs/api`.
