import os
import sys

from dataclasses import dataclass
from pathlib import Path

from src.logging import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from functools import partial

import xgboost

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("Artifacts", "Model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    
    def Build_ANN(self, input_dim):
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


    def intiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the data into training and testing data!!")
            
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                'Linear Regression' : LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(alpha=0.001),
                'Decision Tree Regressor' : DecisionTreeRegressor(),
                'Ada Boost Regressor' : AdaBoostRegressor(),
                'Gradient Boosting Regressor' : GradientBoostingRegressor(),
                'Random Forest Regressor' : RandomForestRegressor(),
                'XGB Regressor' : XGBRegressor(),
                'KNeighbors Regressor' : KNeighborsRegressor(),
                'Artificial Neural Network': KerasRegressor(
                    model=partial(self.Build_ANN, input_dim=X_train.shape[1]),
                    epochs=100,
                    batch_size=32,
                    verbose=0
                )
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            print(model_report)
            
            print("\n===============================================================\n")
            
            logging.info(f"Model Report : {model_report}")
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            print(f"Best Model Found, Model Name : {best_model_name} & R2_score : {best_model_score}")
            
            print("\n===============================================\n")
            
            logging.info(f"Best Model Found, Model Name : {best_model_name} & R2 Score : {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
        except Exception as e:
            logging.info("Error occured during the model training")
            raise CustomException(e,sys)