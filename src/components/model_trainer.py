import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.model_selection import RandomizedSearchCV


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splting Training and Test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'LinearRegression': LinearRegression(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(),
                'AdaBoost Regression': AdaBoostRegressor()
}
            
            # Define parameter grids for each model
            param_grids = {
                'LinearRegression': {
                    'fit_intercept': [True, False]
                },
                'K-Neighbors Regressor': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                'Decision Tree Regressor': {
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'Random Forest Regressor': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'Gradient Boosting Regressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 10]
                },
                'XGB Regressor': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 10]
                },
                'CatBoost Regressor': {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5, 10]
                },
                'AdaBoost Regression': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }

            # for name, model in models.items():
            #     if name in param_grids:
            #         search = RandomizedSearchCV(
            #             model,
            #             param_distributions=param_grids[name],
            #             n_iter=5,
            #             cv=3,
            #             scoring='r2',
            #             n_jobs=-1,
            #             random_state=42
            #         )
                    # search.fit(X_train, y_train)
                    # models[name] = search.best_estimator_  # Replace with best estimator

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test,
                y_test=y_test, models=models, param = param_grids
            )
            

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj= best_model
            )

            predicted = best_model.predict(X_test)

            best_r2_score = r2_score(y_test, predicted)
            return best_r2_score

        except Exception as e:
            raise CustomException(e, sys)

