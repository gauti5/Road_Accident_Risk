import os
import sys

from dataclasses import dataclass
from pathlib import Path

from src.logging import logging
from src.exception import CustomException
from src.utils import save_object

import xgboost

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("Artifacts", "Model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def Build_ANN(ndim, batch_size=None, optimizer='adam', **kwargs):
        model = Sequential()
        model.add(Dense(64, input_dim=ndim.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer, loss='mse')
        return model


    def intiate_model_trainer(self, train_array, test_array):
        pass