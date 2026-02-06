# SLAPRES — Fraud Detection MLOps Pipeline

End-to-end fraud detection system designed to analyze, predict and monitor online payment fraud, from raw data ingestion to model deployment, using modern MLOps practices.

---

## Project Overview

SLAPRES is a complete fraud detection pipeline demonstrating how a machine learning model can be integrated into an industrial-grade architecture.

The project addresses core fraud detection challenges:
- Highly imbalanced datasets
- Need for explainable predictions
- Data drift over time
- Automated retraining and deployment

The system is designed with a strong focus on automation, reproducibility, and scalability.

---

## Key Features

- Automated data ingestion from cloud storage
- Robust preprocessing and feature engineering
- Fraud detection model training with experiment tracking
- Data drift detection
- Conditional retraining orchestration
- REST API for real-time predictions
- CI/CD-ready, production-oriented architecture

---

## Architecture Overview

The project follows a modular MLOps architecture:

1. Data Layer  
   Raw and processed datasets stored on AWS S3

2. Orchestration  
   Apache Airflow DAGs manage ingestion, preprocessing, drift detection and training

3. Monitoring  
   Data drift detection comparing incoming data against a reference dataset

4. ML Lifecycle  
   Training, metrics and artifacts tracked with MLflow  
   Model versioning via a model registry

5. Serving  
   Trained models exposed through a REST API

6. CI/CD  
   Automated testing, build and deployment via GitHub Actions

---

## Pipeline Steps

1. Data ingestion from S3 (RAW zone)
2. Data cleaning and feature engineering
3. Upload of processed data (PROCESSED zone)
4. Data drift detection
5. Conditional model retraining
6. Model registration and deployment
7. Real-time fraud prediction via API

---

## Project Structure

.
├── src/
│   ├── dags/               Airflow DAGs
│   ├── preprocessing/      Data cleaning and feature engineering
│   ├── training/           Model training logic
├── docker/
│   └── docker-compose.yml  Local Airflow setup
├── test/                   Validation and test scripts
├── requirements.txt
├── main.py
└── README.md

---

## Dataset Characteristics

- Approximately 300,000 e-commerce transactions
- Strong class imbalance (around 2 percent fraud)
- Structured tabular format
- Designed for supervised fraud detection

Raw data and generated outputs are intentionally excluded from the repository.

---

## Model and Evaluation

- Supervised classification approach
- Metrics adapted to imbalanced data (precision, recall, F1-score)
- Emphasis on false positive control
- Prediction explainability integrated at application level

---

## Monitoring and Retraining

- Continuous data drift detection
- Automatic retraining triggered when drift is detected
- Full traceability of experiments and model versions

---

## Limitations

- Dataset is static and does not include real-time streaming
- Financial impact estimation is indicative
- Deep learning and graph-based approaches are not implemented

---

## Future Improvements

- Graph-based fraud detection
- Deep learning models
- Dynamic decision thresholds
- Real-time streaming pipeline
- Extended monitoring (concept drift, performance drift)

---

## Authors

Project developed as part of an AI Architect / Data Engineer training program.

Birane Wane  
Sébastien Lartigue  
Selma Rochet  

---

## License

This project is provided for educational and demonstration purposes.
