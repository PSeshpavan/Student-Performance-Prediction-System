# Student Performance Prediction System

An **End-to-End Machine Learning Web Application** designed to predict student academic performance based on demographic and behavioral data. This project demonstrates a production-grade MLOps workflow involving modular code, automated pipelines, experiment tracking, and data versioning.

## 🚀 Key Features

*   **Prediction Pipeline**: A Flask-based web interface for real-time student score predictions.
*   **Automated Training Pipeline**: Orchestrates Data Ingestion, Transformation, Monitoring, and Model Training.
*   **Model Monitoring**: Integrated system to detect **Data Drift** between training and production data using statistical checks.
*   **Experiment Tracking**: Uses **MLflow** (via Dagshub) to log metrics, parameters, and model artifacts.
*   **Data Version Control**: Uses **DVC** to track datasets and ensure reproducibility.
*   **Modular Architecture**: Clean code structure with separate components for Ingestion, Transformation, and Training.

## 🛠️ Tech Stack

*   **Language**: Python 3.8+
*   **Web Framework**: Flask
*   **ML Libraries**: Scikit-learn, XGBoost, CatBoost, Pandas, NumPy
*   **Ops & Tools**: MLflow, DVC, Git, Docker (ready)

## 📂 Project Structure

```
├── artifacts/          # Stores generated files (models, preprocessors, datasets)
├── logs/               # Application and training logs
├── notebook/           # Jupyter notebooks for EDA
├── src/
│   └── my_project/
│       ├── components/ # Core logic (Ingestion, Transformation, Training, Monitoring)
│       ├── pipelines/  # Orchestration scripts (Prediction, Training)
│       ├── logger.py   # Logging configuration
│       └── utils.py    # Utility functions
├── templates/          # HTML templates for Flask
├── app.py              # Flask entry point
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and add your MLflow credentials:
    ```env
    MLFLOW_TRACKING_URI="https://dagshub.com/<username>/DS_Project-1.mlflow"
    MLFLOW_TRACKING_USERNAME="<your_username>"
    MLFLOW_TRACKING_PASSWORD="<your_password>"
    ```

## 🏃‍♂️ Usage

### 1. Run the Web Application
To start the Flask app for predictions:
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

### 2. Run the Training Pipeline
To execute the full training flow (Ingestion -> Monitoring -> Transformation -> Training):
```bash
python -m src.my_project.pipelines.training_pipeline
```
*   **Note**: This pipeline now includes a **Model Monitoring** step that checks for data drift before proceeding to transformation.

## 📊 Modules detailed

*   **Data Ingestion**: Reads from source (SQL/CSV/API), splits into Train/Test, and saves artifacts.
*   **Model Monitoring**: Compares statistical properties (Mean, Std Dev) of the new data against the training baseline to alert on drift.
*   **Data Transformation**: Handles missing values, performs One-Hot Encoding for categorical variables, and scales numerical features.
<<<<<<< HEAD
*   **Model Trainer**: Trains multiple models (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBRegressor, CatBoost, AdaBoost), checks their performance, and saves the best one (threshold: R2 > 0.6).
=======
*   **Model Trainer**: Trains multiple models (Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBRegressor, CatBoost, AdaBoost), checks their performance, and saves the best one (threshold: R2 > 0.6).
>>>>>>> e98317db8d1ef835408323736bdb1508fbff6186
