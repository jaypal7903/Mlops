# 📧 SMS Spam Detection using DVC Pipeline

## 📌 Project Overview

This project demonstrates an end-to-end DVC pipeline for SMS Spam Detection using **Data Version Control (DVC)**. The pipeline automates the complete machine learning workflow, from data ingestion to model evaluation, ensuring reproducibility and modular development.

The project uses the **SMS Spam Collection Dataset (`spam.csv`)**. Text messages are preprocessed and transformed into numerical features using **TF-IDF Vectorization**, followed by training a **Random Forest Classifier**. The processed data and model artifacts are versioned using **DVC**, with **Amazon S3** configured as the remote storage for data versioning.

---

## 🚀 Features

* End-to-end DVC pipeline
* Data versioning with DVC
* Amazon S3 as DVC remote storage
* Modular pipeline architecture
* Text preprocessing using NLTK
* TF-IDF feature extraction
* Random Forest Classifier
* Model evaluation using Accuracy and Precision
* Parameter management using `params.yaml`
* Logging for every pipeline stage
* Git and GitHub integration

---

## ☁️ DVC Remote Storage

The project uses **Amazon S3** as the remote storage backend for DVC.

### Workflow

```text
Local Dataset
      │
      ▼
DVC Tracking
      │
      ▼
Amazon S3 Bucket
      │
      ▼
Versioned Data Storage
```

This allows:

* Versioning of datasets and model artifacts
* Efficient storage of large files outside Git
* Easy collaboration and reproducible experiments

---

## ⚙️ Pipeline Stages

### 1. Data Ingestion

* Loads the `spam.csv` dataset
* Splits the dataset into training and testing sets
* Saves the datasets for downstream stages

### 2. Data Preprocessing

* Removes duplicate records
* Handles missing values
* Encodes target labels
* Saves cleaned datasets

### 3. Feature Engineering

* Converts text to lowercase
* Removes punctuation
* Removes stop words
* Applies stemming
* Generates TF-IDF feature vectors

### 4. Model Building

* Trains a Random Forest Classifier
* Saves the trained model

### 5. Model Evaluation

The model is evaluated using:

* **Accuracy**
* **Precision**

The evaluation results are stored in:

```text
reports/metrics.json
```

---

## 📊 Evaluation Metrics

| Metric    | Description                                                                   |
| --------- | ----------------------------------------------------------------------------- |
| Accuracy  | Measures the overall percentage of correctly classified messages.             |
| Precision | Measures the proportion of messages predicted as spam that are actually spam. |

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* TF-IDF Vectorizer
* RandomForestClassifier
* DVC
* Amazon S3
* Git & GitHub
* YAML
* Python Logging

---

## 🔄 Reproducibility

This project uses DVC to:

* Track datasets and model artifacts
* Reproduce the entire machine learning pipeline using `dvc repro`
* Store versioned data in Amazon S3
* Manage changes in data and pipeline stages efficiently

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearchCV or Optuna
* DVC experiment tracking (`dvc exp`)
* Streamlit or FastAPI deployment
* Docker containerization
* GitHub Actions for CI/CD
* Model monitoring and automated retraining
