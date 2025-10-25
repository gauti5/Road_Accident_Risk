import os
import sys

from src.logging import logging
from src.exception import CustomException

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