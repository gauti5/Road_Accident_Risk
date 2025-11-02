import os
import sys

from src.logging import logging
from src.exception import CustomException

from sklearn.metrics import r2_score

from pathlib import Path

import pickle

def save_object(file_path, obj):
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        logging.info("Error occured during saving the file")
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report={}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train, y_train)
            
            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)
            
            train_model_score=r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]]=test_model_score
            
        return report
    
    except Exception as e:
        logging.info("Error occured during predictions")
        raise CustomException(e,sys)
    
    
def load_object(filepath):
    try:
        
        with open(filepath, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occured while loading object')
        raise CustomException(e,sys)