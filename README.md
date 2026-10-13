# 🛡️ SpamShield ML — End-to-End Spam Detection & MLOps Platform

An end-to-end Machine Learning project that classifies text messages as **spam** or **ham (legitimate)** using NLP and machine learning. The application includes a FastAPI backend, a web dashboard, MLflow experiment tracking, automated tests, Docker support, and a live deployment on Render.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow" alt="MLflow" />
  <img src="https://img.shields.io/badge/Pytest-Testing-0A9EDC?logo=pytest" alt="Pytest" />
  <img src="https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker" alt="Docker" />
  <img src="https://img.shields.io/badge/Deployment-Render-46E3B7?logo=render" alt="Render" />
</p>

## 🚀 Live Demo

- **Live Application:** https://spamshield-ml-project.onrender.com
- **Swagger API Documentation:** https://spamshield-ml-project.onrender.com/docs
- **ReDoc Documentation:** https://spamshield-ml-project.onrender.com/redoc
- **Health Check:** https://spamshield-ml-project.onrender.com/api/health
- **GitHub Repository:** https://github.com/prajapatishubham336/SpamShield-ML-Project

> **Note:** The application is hosted on Render's free instance. It may spin down after inactivity, so the first request after a period of inactivity can take longer.
<img width="1350" height="642" alt="image" src="https://github.com/user-attachments/assets/9892b5e2-ca0e-4b84-9fb8-62db85067358" />

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Installation and Setup](#-installation-and-setup)
- [Model Training](#-model-training)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Running the Application](#-running-the-application)
- [API Documentation](#-api-documentation)
- [Docker](#-docker)
- [Testing](#-testing)
- [Model Evaluation](#-model-evaluation)
- [CI/CD](#-cicd)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Author](#-author)
- [License](#-license)

## 📖 Overview

SpamShield ML is a machine learning application that identifies spam messages and distinguishes them from legitimate messages.

The project combines text preprocessing, TF-IDF feature extraction, machine learning classifiers, and a FastAPI application. A trained pipeline is saved and loaded by the application to make predictions through REST API endpoints and a web interface.

Beyond model training, this project demonstrates practical ML engineering concepts such as experiment tracking, model persistence, automated testing, logging, containerization, and cloud deployment.

## ✨ Key Features

- **Spam Classification:** Classifies messages as `spam` or `ham`.
- **NLP Preprocessing:** Cleans and validates text data before training.
- **TF-IDF:** Converts text into numerical features for machine learning.
- **Multiple Models:** Evaluates Logistic Regression, Multinomial Naive Bayes, and Complement Naive Bayes.
- **Model Selection:** Selects a candidate according to the metrics configured in the training pipeline.
- **Model Persistence:** Saves the trained pipeline using Joblib.
- **FastAPI Backend:** Provides prediction, health, and model-information endpoints.
- **Web Dashboard:** Offers a browser-based interface for interacting with the model.
- **MLflow:** Tracks experiments, parameters, and evaluation metrics.
- **Automated Tests:** Uses pytest to test important application and ML functionality.
- **Docker:** Packages the application and its dependencies into a container.
- **Cloud Deployment:** Runs the application on Render.
- **GitHub Actions:** Provides CI checks for tests and Docker image builds.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| pandas and NumPy | Data handling and numerical operations |
| scikit-learn | Feature extraction and ML models |
| TF-IDF | Text feature representation |
| FastAPI | REST API and application backend |
| Uvicorn | ASGI server |
| MLflow | Experiment tracking |
| Joblib | Model serialization |
| pytest | Automated testing |
| HTML | Web dashboard |
| Docker | Containerization |
| Git and GitHub | Version control and source hosting |
| GitHub Actions | CI automation |
| Render | Cloud hosting |

## 🏗️ System Architecture

```text
             Raw Spam Dataset
                     |
                     v
           Data Preprocessing
                     |
                     v
             Train/Test Split
                     |
                     v
              TF-IDF Features
                     |
                     v
            Candidate ML Models
         /           |           \
        v            v            v
   Logistic       Multinomial   Complement
  Regression     Naive Bayes   Naive Bayes
         \           |           /
                     v
             Model Evaluation
                     |
                     v
               Model Selection
                     |
             +-------+-------+
             |               |
             v               v
        Save Pipeline    MLflow Tracking
             |
             v
          FastAPI
          /      \
         v        v
    Dashboard   REST API
                     |
                     v
              Spam Prediction

GitHub Push --> GitHub Actions --> Tests --> Docker Build
                                            |
                                            v
                                  Deployment workflow
                                  (CI-gated deployment
                                   being configured)
                                            |
                                            v
                                          Render
```

## 📂 Project Structure

```text
SpamShield-ML-Project/
├── .github/
│   └── workflows/
│       └── ci.yml
├── admin/
│   └── app.py
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logging_config.py
│   ├── middleware.py
│   ├── mlops/
│   │   └── mlflow_utils.py
│   ├── monitoring/
│   │   ├── metrics.py
│   │   └── predictions.py
│   ├── pipeline/
│   │   ├── features.py
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   └── training.py
│   └── templates/
│       └── index.html
├── artifacts/
│   └── spam_pipeline.joblib
├── data/
│   ├── raw/
│   │   └── spam.csv
│   ├── processed/
│   └── logs/
├── models/
├── reports/
├── scripts/
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_training.py
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

*This is a representative project structure. Some directories may contain additional files or be empty in a fresh checkout. The deployed application uses the model artifact configured for the application.*

## ⚙️ How It Works

### 1. Data Preprocessing

The preprocessing pipeline prepares the dataset for training. Depending on the implementation, it validates the required columns, normalizes labels, removes missing or blank messages, filters invalid labels, and handles duplicate records.

### 2. TF-IDF Feature Extraction

TF-IDF converts text messages into numerical vectors so that machine learning algorithms can process them.

The documented feature configuration includes:

- Lowercase normalization
- Unicode accent stripping
- Unigrams and bigrams
- Maximum vocabulary size of 20,000 features
- Sublinear term frequency scaling

### 3. Model Training

The training pipeline evaluates three candidate algorithms:

- **Logistic Regression:** A linear classification model.
- **Multinomial Naive Bayes:** A probabilistic classifier commonly used for text classification.
- **Complement Naive Bayes:** A Naive Bayes variant that can be useful for imbalanced text datasets.

### 4. Model Evaluation and Selection

The training pipeline evaluates candidate models using configured metrics, which include accuracy, spam precision, spam F1-score, and macro F1-score.

The selected pipeline is serialized and saved for inference. MLflow records experiment information according to the project's configuration.

### 5. Prediction

The FastAPI application loads the saved pipeline and uses it to classify incoming text. Prediction responses can include a label, spam indicator, confidence, and spam probability.

Confidence and probability values are model outputs, not guarantees that a prediction is correct.

## 🚀 Installation and Setup

### Prerequisites

- Python 3.11
- Git
- Anaconda or Miniconda (optional)
- Docker Desktop (optional, for containerized execution)

### 1. Clone the Repository

```powershell
git clone https://github.com/prajapatishubham336/SpamShield-ML-Project.git
cd SpamShield-ML-Project
```

### 2. Create a Python Environment

Using Conda:

```powershell
conda create -n llmapp python=3.11 -y
conda activate llmapp
```

Alternatively, use a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

If the application requires environment variables, copy the example file:

```powershell
Copy-Item .env.example .env
```

Configure only the variables required by the application. Never commit secrets or your actual `.env` file.

### 5. Verify the Dataset

Check whether the training dataset is present at:

```text
data/raw/spam.csv
```

If the dataset is unavailable, obtain or generate it according to the dataset requirements implemented in the training pipeline.

## 🧠 Model Training

Run the training module from the repository root:

```powershell
python -m app.pipeline.training
```

The pipeline is intended to:

1. Load the raw dataset.
2. Preprocess text and labels.
3. Split the data into training and testing sets.
4. Train and evaluate candidate models.
5. Log experiment information to MLflow.
6. Select a model according to configured evaluation metrics.
7. Save the trained pipeline.
8. Generate evaluation reports where configured.

The current documented training configuration uses a test size of `0.2` and a random state of `42`.

The application artifact is expected at:

```text
artifacts/spam_pipeline.joblib
```

Check the training configuration before retraining to confirm the exact output path and whether any additional model location is used.

## 📊 MLflow Experiment Tracking

MLflow is used to record model training experiments, parameters, and evaluation metrics.

To start the MLflow UI, run the following command from the project directory:

```powershell
mlflow ui --host 127.0.0.1 --port 5000 --workers 1
```

Open:

http://127.0.0.1:5000

If your configuration uses a local SQLite tracking database, ensure the MLflow command points to the correct database URI. A local tracking database is a development resource and is not automatically shared with the deployed Render service.

## 🌐 Running the Application

Start the FastAPI server from the project root:

```powershell
uvicorn app.main:app --reload
```

Open the following local URLs:

| Resource | URL |
|---|---|
| Web Dashboard | http://127.0.0.1:8000/ |
| Swagger API Documentation | http://127.0.0.1:8000/docs |
| ReDoc Documentation | http://127.0.0.1:8000/redoc |
| Health Check | http://127.0.0.1:8000/api/health |
| MLflow UI | http://127.0.0.1:5000 |

The MLflow UI runs as a separate process.

## 🔌 API Documentation

Interactive documentation is available at:

**Live Swagger:** https://spamshield-ml-project.onrender.com/docs

### Main Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Web dashboard |
| GET | `/api/health` | Health check |
| GET | `/health` | Health check, if enabled |
| POST | `/predict` | Single-message prediction |
| POST | `/api/predict` | Single-message prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/api/model-info` | Model information |

Refer to the live Swagger documentation for the exact request schemas and response fields.

### Example: Single Prediction

**Request**

```http
POST /predict
Content-Type: application/json
```

```json
{
  "text": "Congratulations! You have won a free prize. Claim now!"
}
```

Example using PowerShell against the deployed service:

```powershell
$body = @{
    text = "Congratulations! You have won a free prize. Claim now!"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "https://spamshield-ml-project.onrender.com/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

A successful response may contain fields such as:

```json
{
  "label": "spam",
  "prediction": "spam",
  "is_spam": true,
  "confidence": 0.95,
  "spam_probability": 0.95,
  "message": "Prediction completed successfully."
}
```

This is an illustrative example only. Actual predictions and confidence values depend on the deployed model and input message.

### Health Check

```powershell
Invoke-RestMethod `
    -Uri "https://spamshield-ml-project.onrender.com/api/health"
```

A healthy response should indicate that the application is running and the model is loaded.

### Batch Prediction

```json
{
  "texts": [
    "Congratulations! Claim your reward now!",
    "Hi, are we still meeting today?"
  ]
}
```

Send this JSON to `POST /predict/batch`. Refer to Swagger for the exact response structure.

## 🐳 Docker

Build the Docker image:

```powershell
docker build -t spamshield-ml .
```

Run the container locally:

```powershell
docker run --rm -p 8000:8000 spamshield-ml
```

Open:

http://localhost:8000/docs

The Docker image packages the application and its dependencies. The container startup command supports the `PORT` environment variable used by Render.

## 🧪 Testing

Run the automated tests:

```powershell
python -m pytest -v
```

The test suite covers areas including API behavior, preprocessing, and training functionality.

A previous local run reported **9 passed, 4 warnings**. This is a historical result, not a guarantee that every future run will produce the same result.

Review warnings when upgrading dependencies or modifying the application.

## 📈 Model Evaluation

Previously recorded evaluation metrics:

| Metric | Observed Value |
|---|---:|
| Accuracy | 0.750 |
| Spam Precision | 0.667 |
| Spam Recall | 1.000 |
| Spam F1-score | 0.800 |
| Macro F1-score | 0.733 |

These values describe the recorded evaluation run and should not be treated as verified production performance.

The sample dataset contains duplicate messages and a limited number of unique examples. This can make evaluation results overly optimistic if similar messages appear in both training and testing data.

For a more reliable evaluation, use a larger and more diverse dataset, investigate duplicate and near-duplicate leakage, and report per-class metrics and a confusion matrix.

## 🔄 CI/CD

The project includes a GitHub Actions workflow at `.github/workflows/ci.yml`.

The CI workflow is designed to:

1. Run when changes are pushed to `main` or a pull request targets `main`.
2. Set up Python 3.11.
3. Install dependencies.
4. Run the pytest test suite.
5. Build the Docker image.

**Deployment automation status:** The application is already deployed on Render. Automatic deployment gated on successful GitHub Actions checks is a separate configuration step and should be marked complete only after a successful end-to-end workflow run.

Repository: https://github.com/prajapatishubham336/SpamShield-ML-Project

## ⚙️ Configuration

Important configuration areas include:

- Model artifact path
- Spam classification threshold
- Dataset paths
- MLflow tracking URI
- MLflow experiment name
- Logging configuration

For implementation details, refer to:

- `app/config.py`
- `app/mlops/mlflow_utils.py`
- `app/pipeline/training.py`

The exact configuration depends on the current code and environment variables.

## 🔮 Limitations and Future Improvements

### Current Limitations

- The sample dataset is small and contains duplicate messages.
- Existing evaluation metrics are insufficient to establish real-world performance.
- The model can produce false positives and false negatives.
- Local MLflow tracking is not automatically a shared production tracking service.
- The free Render instance can spin down after inactivity.

### Planned Improvements

- [ ] Use a larger and more diverse spam dataset.
- [ ] Improve data validation and prevent train/test leakage.
- [ ] Add confusion matrix and per-class evaluation reports.
- [ ] Improve structured logging and prediction monitoring.
- [ ] Add linting and additional automated quality checks.
- [ ] Verify GitHub Actions CI on every relevant change.
- [ ] Gate Render deployment on successful CI checks.
- [ ] Improve model versioning and reproducibility.
- [ ] Enhance the web dashboard and prediction experience.

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

- NLP text preprocessing and TF-IDF
- Training and comparing machine learning models
- Model evaluation and serialization
- Building REST APIs with FastAPI
- Experiment tracking with MLflow
- Automated testing using pytest
- Containerizing applications with Docker
- Version control with Git and GitHub
- Deploying a machine learning application on Render
- Developing a CI/CD workflow using GitHub Actions

## 👨‍💻 Author

**Shubham Prajapati**

- GitHub: https://github.com/prajapatishubham336
- LinkedIn: https://www.linkedin.com/in/shubham-prajapati-103b38393/

## 📄 License

No license has been specified yet. If you intend to publish this repository as open source, add an appropriate `LICENSE` file and verify that the dataset and third-party resources permit your intended use.
