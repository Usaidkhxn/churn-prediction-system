# Churn Prediction & Retention Targeting System

An end-to-end machine learning system for predicting customer churn and optimizing retention decisions using expected-value based targeting.  
Designed with production-style architecture, batch and real-time inference, and business-aware decision logic.

---

## 🚀 Key Features

- End-to-end ML pipeline: data ingestion → preprocessing → training → evaluation → deployment
- Mixed-type preprocessing using `ColumnTransformer` (numeric + categorical features)
- Imbalanced churn classification with Logistic Regression baseline
- Model evaluation using ROC-AUC, PR-AUC, F1, precision, recall, and confusion matrix
- Business decision layer for **budget-aware retention targeting**
- FastAPI service for real-time and batch inference
- CLI tool for large-scale offline batch scoring

---

## 📊 Model Performance (Baseline)

| Metric | Value |
|------|------|
| ROC-AUC | **0.84** |
| PR-AUC | **0.63** |
| Recall (Churners) | **0.76** |
| Accuracy | 0.75 |

---

## 🧠 Business Logic: Retention Targeting

Customers are prioritized using an expected value (EV) framework:

```text
EV = P(churn) × P(retention_success) × value_saved − cost_of_offer

```

This allows targeting only customers who provide **positive expected business value** under a fixed budget.

---

## 🏗 Project Architecture

```text
data/raw
↓
preprocessing (impute + encode + scale)
↓
model training & evaluation
↓
artifacts (model + metrics)
↓
FastAPI service ←→ CLI batch scoring
```

## 📁 Repository Structure

```text
churn-ml-system/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── model.joblib
│   └── metrics.json
├── src/
│   └── ml_system/
│       ├── api.py
│       ├── cli_score.py
│       ├── config.py
│       ├── data.py
│       ├── decision.py
│       ├── features.py
│       ├── model.py
│       └── train.py
└── README.md

```

## ⚙️ Setup & Installation

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

```
## 🏋️ Train the Model

```text
python -m src.ml_system.train

```

## 🌐 Run the API

```text
uvicorn src.ml_system.api:app --reload

```

## Swagger UI:

```text

http://127.0.0.1:8000/docs

```

## 🧪 Batch Scoring (CLI)

```text

python -m src.ml_system.cli_score \
  --input data/raw/churn.csv \
  --output artifacts/scored_churn.csv

```
### 📌 Technologies



- Python, Pandas, NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Joblib
- YAML configuration


