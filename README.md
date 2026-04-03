# 🚀 Automated End-to-End MLOps System

A **production-ready AutoML + MLOps platform** that enables users to upload structured datasets, automatically train and select the best machine learning model, track experiments, and deploy predictions via API.

---

## 📌 Overview

This system eliminates manual ML workflows by providing a **fully automated pipeline**:

**Data → Preprocessing → Model Training → Model Selection → Tracking → Deployment → Prediction**

It is designed to simulate **real-world ML systems used in industry**.

---

## ✨ Key Features

* 🔄 **Automated Preprocessing**

  * Missing value handling
  * Categorical encoding (One-Hot / Label Encoding)
  * Feature scaling

* 🧠 **Automatic Problem Detection**

  * Classification vs Regression based on target column

* 🤖 **Model Training & Selection**

  * Trains multiple models:

    * Random Forest
    * Gradient Boosting
    * Logistic / Linear Regression
  * Selects best model using performance metrics

* 📊 **Experiment Tracking**

  * Integrated with **MLflow**
  * Logs parameters, metrics, and model versions

* 🌐 **API Deployment**

  * Built with **FastAPI**
  * Endpoints:

    * `/train` → Train model
    * `/predict` → Generate predictions

* 🐳 **Dockerized System**

  * Fully containerized using Docker & Docker Compose

* ⚙️ **CI/CD Ready**

  * GitHub Actions for testing and build validation

---

## 🏗️ Project Structure

```bash
├── api/               # FastAPI endpoints
├── pipelines/        # Training pipeline
├── models/           # Model logic
├── utils/            # Preprocessing & helpers
├── tests/            # Unit tests
├── data/             # Sample datasets
├── uploads/          # Uploaded datasets (ignored in git)
├── models_store/     # Saved models (ignored in git)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 🔹 Option 1: Run with Docker (Recommended)

```bash
docker-compose up --build
```

Access:

* API → http://localhost:8000
* MLflow → http://localhost:5000

---

### 🔹 Option 2: Run Locally

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Start MLflow server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

#### 3. Start FastAPI

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🚀 Usage Guide

### 🔹 Step 1: Train Model

* Open → http://localhost:8000
* Upload CSV dataset
* Enter target column (e.g., `price`)
* Click **Train Model**

---

### 🔹 Step 2: Predict

Example input:

```json
{
  "area": 1600,
  "bedrooms": 3
}
```

👉 Output:

```json
{
  "prediction": 5000000
}
```

---

## 📊 MLflow Tracking

* View experiments at → http://localhost:5000
* Track:

  * model performance
  * parameters
  * best model

---

## 🧪 Testing

Run tests using:

```bash
pytest tests/
```

---

## 🧠 Tech Stack

* Python
* Scikit-learn
* FastAPI
* MLflow
* Docker
* GitHub Actions

---

## 🎯 Key Highlights

* End-to-End ML pipeline
* AutoML-like functionality
* Production-ready architecture
* API + Deployment ready
* Scalable & modular design

---

## 📌 Future Improvements

* Model registry (production/staging)
* UI enhancements
* Cloud deployment (AWS/GCP)
* Advanced feature engineering
* Monitoring & drift detection

---

## 👨‍💻 Author

**Dharmik Dudhat**

---

⭐ If you found this project useful, consider giving it a star!
