import sys
import os
import pandas as pd
import numpy as np
from src.my_project.logger import logging
from src.my_project.exception import CustomException

class ModelMonitoring:
    def __init__(self):
        pass

    def initiate_model_monitoring(self, train_data_path: str, test_data_path: str):
        """
        Basic monitoring to check for data drift between training (reference) 
        and testing (current) datasets using descriptive statistics.
        """
        try:
            logging.info("Initiating model monitoring - Data Drift Check")
            
            # Load datasets
            if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
                raise FileNotFoundError("Reference or Current data file not found.")
                
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info(f"Loaded Reference Data (Rows: {len(train_df)}) and Current Data (Rows: {len(test_df)})")
            
            # Identify numerical columns for drift check
            numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
            
            drift_report = {}
            drift_detected = False
            
            for feature in numerical_features:
                # Calculate basic stats
                train_mean = train_df[feature].mean()
                train_std = train_df[feature].std()
                test_mean = test_df[feature].mean()
                test_std = test_df[feature].std()
                
                # Simple Drift Threshold (e.g., if mean shifts by more than 1 standard deviation)
                # In production, use more robust tests like KS-test or PSI (Population Stability Index)
                threshold = train_std * 1.0 
                diff = abs(train_mean - test_mean)
                
                is_drift = diff > threshold
                
                drift_report[feature] = {
                    "train_mean": round(train_mean, 3),
                    "test_mean": round(test_mean, 3),
                    "diff": round(diff, 3),
                    "threshold": round(threshold, 3),
                    "drift_detected": is_drift
                }
                
                if is_drift:
                    drift_detected = True
                    logging.warning(f"Drift DETECTED in feature '{feature}'. Diff: {diff:.3f} > Threshold: {threshold:.3f}")
                else:
                    logging.info(f"No drift in feature '{feature}'.")

            logging.info("Model Monitoring Completed. Drift Report generated.")
            
            # You could save this report to a file or database here
            
            return drift_report

        except Exception as e:
            raise CustomException(e, sys)
