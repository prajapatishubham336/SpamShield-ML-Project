# 🛡️ SpamShield ML — End-to-End Spam Detection & MLOps Platform

**An end-to-end Machine Learning project that classifies messages as spam or ham (legitimate messages), with a FastAPI backend, automated model selection, MLflow experiment tracking, and a web interface.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow" alt="MLflow" />
  <img src="https://img.shields.io/badge/Pytest-Testing-0A9EDC?logo=pytest" alt="Pytest" />
  <img src="https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker" alt="Docker" />
</p>

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
- [Testing](#-testing)
- [Model Evaluation](#-model-evaluation)
- [Configuration](#-configuration)
- [Limitations and Future Improvements](#-limitations-and-future-improvements)
- [Learning Outcomes](#-learning-outcomes)
- [License](#-license)

## 📖 Overview

SpamShield ML is a machine learning application designed to identify spam messages and distinguish them from legitimate messages.

The project combines a traditional NLP-based classification pipeline with a FastAPI application and MLflow experiment tracking. It demonstrates how a machine learning model can be trained, evaluated, saved, loaded into an API, and used to make predictions through a web interface.

The application uses text preprocessing, TF-IDF feature extraction, and multiple machine learning algorithms to build a spam detection pipeline.

Rather than focusing only on model training, SpamShield demonstrates important parts of a practical ML engineering workflow, including API development, automated testing, model persistence, logging, and experiment tracking.

## ✨ Key Features

- **Text Classification:** Predicts whether a message is `spam` or `ham`.
- **Text Preprocessing:** Cleans and prepares the dataset before training.
- **TF-IDF Feature Extraction:** Converts text messages into numerical features.
- **Multiple ML Models:** Evaluates Logistic Regression, Multinomial Naive Bayes, and Complement Naive Bayes.
- **Automated Model Selection:** Selects a candidate model using the evaluation metrics defined in the training pipeline.
- **Model Persistence:** Saves the trained pipeline for reuse during inference.
- **FastAPI Backend:** Exposes prediction and model information through API endpoints.
- **Web Interface:** Provides a simple dashboard for interacting with the application.
- **MLflow Tracking:** Records model training runs, parameters, and evaluation metrics.
- **Prediction Logging:** Stores prediction-related information for monitoring.
- **Automated Testing:** Uses pytest to test API behavior, preprocessing, and training.
- **Docker Support:** Includes a Dockerfile for containerization.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Main programming language |
| pandas | Data handling and preprocessing |
| NumPy | Numerical operations |
| scikit-learn | Feature extraction and ML models |
| TF-IDF | Text feature representation |
| FastAPI | Backend API |
| Uvicorn | ASGI application server |
| MLflow | Experiment tracking |
| Joblib | Model serialization and loading |
| pytest | Automated testing |
| HTML | Web dashboard |
| Docker | Application containerization |
| Git and GitHub | Version control and source hosting |

## 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │   Raw Spam Dataset  │
                 │      spam.csv       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Data Preprocessing │
                 │ Clean, Validate,    │
                 │ Remove Duplicates   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Train / Test Split  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  TF-IDF Features    │
                 │  Unigrams + Bigrams │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Candidate Models    │
                 │ Logistic Regression │
                 │ Multinomial NB      │
                 │ Complement NB       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Evaluate & Select   │
                 │ Best Candidate      │
                 └──────────┬──────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
        ┌─────────────────┐  ┌──────────────────┐
        │ Save ML Pipeline│  │ MLflow Tracking  │
        │ Joblib Artifact │  │ Parameters/Metrics│
        └────────┬────────┘  └──────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │     FastAPI App     │
        │  Load Saved Model   │
        └──────────┬──────────┘
                   │
           ┌───────┴────────┐
           ▼                ▼
  ┌────────────────┐  ┌────────────────┐
  │ Web Dashboard  │  │ REST API       │
  │ User Interface │  │ Predictions    │
  └────────────────┘  └────────────────┘
```

## 📂 Project Structure

The following structure reflects the main directories and files in the project.

```text
SpamShield-ML-Project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── admin/
│   └── app.py
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logging_config.py
│   ├── middleware.py
│   │
│   ├── mlops/
│   │   └── mlflow_utils.py
│   │
│   ├── monitoring/
│   │   ├── metrics.py
│   │   └── predictions.py
│   │
│   ├── pipeline/
│   │   ├── features.py
│   │   ├── models.py
│   │   ├── preprocessing.py
│   │   └── training.py
│   │
│   └── templates/
│       └── index.html
│
├── artifacts/
│   ├── .gitkeep
│   └── spam_pipeline.joblib
│
├── data/
│   ├── raw/
│   │   └── spam.csv
│   ├── processed/
│   └── logs/
│
├── models/
│   └── spam_pipeline.joblib
│
├── reports/
│   └── metrics.json
│
├── scripts/
│   └── make_sample_data.py
│
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_training.py
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── runtime.txt
└── README.md
```

**Note:** Local-only files such as `.env`, `mlflow.db`, logs, caches, and generated artifacts may be excluded from Git using `.gitignore`. The exact files tracked in the repository depend on your Git configuration.

## ⚙️ How It Works

### 1. Data Preprocessing

The preprocessing module prepares the input dataset for machine learning.

It handles tasks such as:

- Checking the required text and label columns.
- Normalizing labels into `spam` and `ham`.
- Removing missing or blank messages.
- Filtering invalid labels.
- Removing duplicate text-label pairs.

### 2. Feature Extraction with TF-IDF

Machine learning algorithms require numerical input. SpamShield uses `TfidfVectorizer` to convert text messages into numerical feature vectors.

The current feature configuration includes:

- Lowercase text normalization.
- Unicode accent stripping.
- Unigrams and bigrams using `ngram_range=(1, 2)`.
- A maximum vocabulary size of 20,000 features.
- Sublinear term frequency scaling.

For example:

```text
Input:
"URGENT! Claim your cash reward now!"

          │
          ▼
     TF-IDF Vectorizer

          │
          ▼
 Numerical Feature Vector

          │
          ▼
      ML Classifier

          │
          ▼
     Spam Prediction
```

### 3. Candidate Model Training

The training pipeline evaluates three candidate algorithms:

**Logistic Regression**

A linear classification algorithm that estimates the probability of a message belonging to a class.

**Multinomial Naive Bayes**

A probabilistic classifier commonly used for text classification and word-frequency-based features.

**Complement Naive Bayes**

A Naive Bayes variant designed to improve classification behavior on imbalanced datasets.

### 4. Model Selection

The training pipeline evaluates candidate models using metrics such as:

- Macro F1-score
- Spam F1-score
- Spam precision
- Accuracy

The selection logic compares these metrics in the order configured in the training code. The selected pipeline is then saved for later use.

If two models have identical scores, the implementation's tie-breaking behavior determines which model is selected.

### 5. Model Inference

When a user submits a message, the FastAPI application passes the text through the saved pipeline and returns the prediction and associated probability information when supported by the model.

## 🚀 Installation and Setup

### Prerequisites

Install the following before starting:

- Python 3.11
- Git
- Anaconda or Miniconda (optional, for environment management)

### Step 1: Clone the Repository


```bash

git clone https://github.com/prajapatishubham336/SpamShield-ML-Project.git
cd SpamShield-ML-Project
```

### Step 2: Create a Conda Environment

```bash
conda create -n llmapp python=3.11 -y
conda activate llmapp
```

Alternatively, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project uses additional development dependencies defined in `pyproject.toml`, install those according to your configuration.

### Step 4: Configure Environment Variables

Copy `.env.example` to `.env` if the application requires local environment variables.

```powershell
Copy-Item .env.example .env
```

Edit `.env` only with values required by your configuration.

**Never commit `.env` or secrets to GitHub.**

### Step 5: Check the Dataset

Ensure the dataset is available at:

```text
data/raw/spam.csv
```

The expected data contains text messages and their labels. The preprocessing pipeline supports the column names and label formats implemented in the project.

If the dataset is not included in your Git repository, add instructions explaining how to obtain or generate it.

## 🧠 Model Training

Run the training module from the project root:

```bash
python -m app.pipeline.training
```

The training process performs the following tasks:

1. Loads the raw dataset.
2. Preprocesses the text and labels.
3. Splits the data into training and testing sets.
4. Builds and evaluates the candidate models.
5. Logs experiment information to MLflow.
6. Selects the best candidate according to the configured metrics.
7. Saves the selected model pipeline.
8. Generates a metrics report.

The training pipeline uses a test size of `0.2` and a random state of `42` in its current configuration.

The saved pipeline is available at:

```text
artifacts/spam_pipeline.joblib
```

The project also contains a model location under `models/`. Check the training configuration to determine whether both locations are actively used.

## 📊 MLflow Experiment Tracking

MLflow records the training experiments and their evaluation metrics.

### Start the MLflow UI

The current project configuration uses a local SQLite tracking database named `mlflow.db`.

Run the following command from a separate terminal in the project directory:

```bash
mlflow ui --backend-store-uri "sqlite:///C:/Users/Admin/Desktop/SpamShield-ML-Project/mlflow.db" --host 127.0.0.1 --port 5000 --workers 1
```

Open:

**http://127.0.0.1:5000**

### What You Can Track

- Experiment name: `SpamShield-ML`
- Candidate model names
- Model training parameters
- Accuracy
- Spam precision
- Spam recall
- Spam F1-score
- Macro F1-score
- Run status and duration

The current training configuration logs experiment runs for the candidate models.

MLflow tracks the results; the model selection decision is performed by the project's training code.

**Important:** The SQLite database is a local development resource. If you want to publish the experiment history, consider exporting the metrics or configuring a shared MLflow tracking server.

## 🌐 Running the Application

Start the FastAPI server from the project root:

```bash
uvicorn app.main:app --reload
```

The application should be accessible at:

| Resource | URL |
|---|---|
| Web Dashboard | http://127.0.0.1:8000/ |
| Swagger API Documentation | http://127.0.0.1:8000/docs |
| ReDoc API Documentation | http://127.0.0.1:8000/redoc |
| Health Endpoint | http://127.0.0.1:8000/api/health |
| MLflow UI | http://127.0.0.1:5000 |

The MLflow UI requires its own process to be running.

## 🔌 API Documentation

SpamShield exposes REST endpoints through FastAPI.

### Main Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Serves the web dashboard |
| GET | `/api/health` | Health check |
| GET | `/health` | Health check |
| POST | `/predict` | Predict a single message |
| POST | `/api/predict` | Prediction endpoint |
| POST | `/predict/batch` | Batch message prediction |
| GET | `/api/model-info` | Model information |

For the exact request schemas and response fields, open the Swagger documentation at `http://127.0.0.1:8000/docs`.

### Example: Single Message Prediction

The following is an illustrative request. Adjust the JSON field names to match the request schema defined in `app/main.py`.

```json
{
  "text": "Congratulations! You have won a free prize. Claim now!"
}
```

Possible classification labels:

- `spam` — the message is classified as spam.
- `ham` — the message is classified as legitimate.

The prediction response may include fields such as:

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

**Note:** The response above is an illustrative example, not a guaranteed live prediction or fixed confidence score. Actual output depends on the saved model, input message, and configured threshold.

### Test Through Swagger

1. Start the FastAPI server.
2. Open `http://127.0.0.1:8000/docs`.
3. Expand the `/predict` endpoint.
4. Click **Try it out**.
5. Enter a message using the schema shown by Swagger.
6. Click **Execute**.
7. Review the prediction and HTTP response.

## 🧪 Testing

Run the automated test suite:

```bash
python -m pytest -q
```

The current project test suite has previously reported:

```text
9 passed, 4 warnings
```

This result represents the test run at that point in development; results may differ after code or dependency changes.

The tests cover areas including:

- API behavior
- Data preprocessing
- Model training and artifact creation

Warnings related to dependency deprecations or dtype conversion do not necessarily mean a test failed. Review them when upgrading dependencies or maintaining the project.

## 📈 Model Evaluation

The MLflow runs currently show the following reported metrics:

| Metric | Observed Value |
|---|---:|
| Accuracy | 0.750 |
| Spam Precision | 0.667 |
| Spam Recall | 1.000 |
| Spam F1-score | 0.800 |
| Macro F1-score | 0.733 |

These values describe the recorded evaluation runs and should not be interpreted as reliable estimates of real-world performance.

The current sample dataset contains many duplicate messages and only a small number of unique examples. As a result, the evaluation may not generalize to unseen messages.

Before deploying this model for real-world use, evaluate it using a larger, more diverse dataset and ensure that duplicate or near-duplicate messages do not leak between training and testing data.

## ⚙️ Configuration

Application settings are managed through the project's configuration modules and environment variables.

Relevant configuration includes:

- Model artifact path
- Spam classification threshold
- MLflow tracking URI
- MLflow experiment name
- Logging settings
- Dataset paths

Example MLflow configuration values used by the project:

```text
MLFLOW_EXPERIMENT_NAME=SpamShield-ML
```

The tracking URI can use a local SQLite database during development. Refer to `app/config.py`, `app/mlops/mlflow_utils.py`, and the training module for the actual configuration behavior.

## 🔮 Limitations and Future Improvements

### Current Limitations

- The sample dataset is small and contains duplicate messages.
- The recorded test metrics are not sufficient to establish production-level accuracy.
- The model may misclassify legitimate messages as spam or miss spam messages.
- MLflow artifacts may need additional logging configuration.
- A local SQLite tracking database is not a shared production tracking service.
- The application has not been demonstrated here to be deployed publicly.

### Planned Improvements

- [ ] Collect a larger and more diverse labeled spam dataset.
- [ ] Improve data validation and prevent train/test data leakage.
- [ ] Add confusion matrix and per-class evaluation reports.
- [ ] Log the trained model and metrics report as MLflow artifacts.
- [ ] Improve structured logging and prediction monitoring.
- [ ] Add CI checks for linting and automated tests.
- [ ] Build and verify the Docker image.
- [ ] Deploy the FastAPI application to a hosting platform.
- [ ] Add model versioning and a reproducible model promotion workflow.
- [ ] Improve the web interface and display prediction results clearly.

## 🎯 Learning Outcomes

By building this project, you can practice:

- Developing a complete machine learning pipeline.
- Cleaning and validating text datasets.
- Applying TF-IDF to NLP classification.
- Training and comparing multiple ML algorithms.
- Selecting and serializing a trained model.
- Building REST APIs using FastAPI.
- Tracking ML experiments with MLflow.
- Testing API endpoints and training workflows with pytest.
- Organizing an ML engineering project for GitHub.
- Preparing a machine learning application for containerization and deployment.

## 📄 License

No license has been specified yet. If you plan to publish this project as open source, add an appropriate `LICENSE` file and confirm that the dataset and any third-party resources permit the intended use.

---

## 👨‍💻 Author

**Your Name**

- GitHub: `https://github.com/prajapatishubham336`
- LinkedIn: [Shubham Prajapati](https://www.linkedin.com/in/shubham-prajapati-103b38393/)

---
