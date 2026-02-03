import sys
import os
from dataclasses import dataclass
import numpy as np
from urllib.parse import urlparse
import mlflow

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from src.my_project.logger import logging
from src.my_project.exception import CustomException
from src.my_project.utils import (
    save_object,
    evaluate_models,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure MLflow directly without dagshub
# Set the tracking URI directly to your DagsHub MLflow instance
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set these environment variables for authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")


@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Initiating Model Trainer")
            logging.info("Split Train and Test Data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            logging.info("Model Training Started")
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info("Model Training Completed")
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            
            best_model = models[best_model_name]
            
            best_params = params[best_model_name]
            
            # MLflow tracking - with error handling
            try:
                logging.info("Starting MLflow logging")
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                
                with mlflow.start_run():
                    predicted_qualities = best_model.predict(X_test)
                    (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)
                    
                    # Log parameters and metrics
                    mlflow.log_params(best_params)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)
                    
                    # Create model signature
                    signature = mlflow.models.infer_signature(
                        X_test,
                        predicted_qualities
                    )
                    
                    # Log the model
                    mlflow.sklearn.log_model(
                        best_model,
                        "model",
                        registered_model_name=best_model_name,  # Use the actual model name directly
                        signature=signature,
                        input_example=X_test[:5]
                    )
                    
                logging.info("MLflow logging completed successfully")
                    
            except Exception as mlflow_error:
                logging.warning(f"MLflow tracking failed: {str(mlflow_error)}")
                logging.warning("Continuing without MLflow tracking")
            
            # Continue with the rest of the process regardless of MLflow status
            if best_model_score < 0.6:
                raise CustomException("No best Model Found", sys)
            
            logging.info(f"Best Model Found, Model Name: {best_model_name}, Model Score: {best_model_score}")
            logging.info("Saving Best Model")
            
            save_object(file_path=self.model_trainer_config.trained_model_filepath, obj=best_model)
            logging.info("Model Saved Successfully")
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            
            return r2
            
        except Exception as e:
            logging.error(f"Exception occurred: {str(e)}")
            raise CustomException(e, sys)