import os
import sys
from src.my_project.logger import logging
from src.my_project.exception import CustomException
from src.my_project.components.data_ingestion import DataIngestion
from src.my_project.components.data_transformation import DataTransformation
from src.my_project.components.model_trainer import ModelTrainer

from src.my_project.components.model_monitoring import ModelMonitoring

class TrainingPipeline:
    def __init__(self):
        pass

    def start_training_pipeline(self):
        logging.info("Entered the training pipeline execution")
        try:
            # Step 1: Data Ingestion
            logging.info("Step 1: Data Ingestion started")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed. Train: {train_data_path}, Test: {test_data_path}")

            # Step 2: Model Monitoring (Data Drift Check)
            # Checking drift between Train (Reference) and Test (Current/Evaluation)
            logging.info("Step 2: Model Monitoring started")
            model_monitoring = ModelMonitoring()
            model_monitoring.initiate_model_monitoring(train_data_path, test_data_path)
            logging.info("Model Monitoring completed")

            # Step 3: Data Transformation
            logging.info("Step 3: Data Transformation started")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_obj_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Data Transformation completed")

            # Step 4: Model Training
            logging.info("Step 4: Model Training started")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model Training completed. R2 Score: {r2_score}")

            return r2_score

        except Exception as e:
            logging.error("Error occurred in training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.start_training_pipeline()
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        print(f"Error: {e}")
