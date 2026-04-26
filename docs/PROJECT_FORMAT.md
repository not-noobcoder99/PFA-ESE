# Project Formatting Notes

This document describes how assignment work was transformed into a semester-project format.

## Migration Map

- Assignment 1 source -> `src/phase1`
- Assignment 2 source -> `src/phase2`
- Assignment 3 API source -> `src/api`
- Shared dataset -> `data/heart.csv`
- Serving model -> `models/RandomForest.joblib`
- Test suites -> `tests/phase2` and `tests/api`
- Generated artifacts -> `outputs/phase1`, `outputs/phase2`, `outputs/api`

## Engineering Improvements Added

- Root-level dependency management (`requirements.txt`)
- Root-level test configuration (`pyproject.toml`)
- Standard project scripts (`scripts/*.ps1`)
- CI pipeline (`.github/workflows/ci.yml`)
- API containerization (`Dockerfile`)
- Clean root project documentation (`README.md`)
