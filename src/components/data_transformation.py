import os
import sys

from src.logging import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass 

@dataclass
class Data_Transformation_config:
    preperocessor_file_path:str=os.path.join("Artifacts", "preprocessor.pkl")
    
class DataTransformation:
    def _init__(self):
        self.data_transformation_config=Data_Transformation_config()
        
    def get_data_transformation(self):
        try:
            
            logging.info("Data Transformation Started!!!")
            
            cat_columns=['road_type', 'lighting', 'weather', 'time_of_day']
            num_columns=['num_lanes','curvature','speed_limit','road_signs_present','public_road','holiday','school_season','num_reported_accidents']
            
            road_type_categories=['highway', 'rural', 'urban']
            lighting_categories=['dim', 'daylight', 'night']
            weather_categories=['foggy', 'clear', 'rainy']
            time_of_day_categories=['morning', 'evening', 'afternoon']
            
            logging.info("Pipeline Started!!!")
            
            Cat_Pipeline=Pipeline(
                steps=[
                    ('encoder', OneHotEncoder(categories=[road_type_categories, lighting_categories, weather_categories, time_of_day_categories], handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            Num_Pipeline=Pipeline(
                steps=[
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ('num pipeline', Num_Pipeline, num_columns),
                    ('Cat Pipeline', Cat_Pipeline, cat_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            logging.info("Exception occured during the data transformation")
            raise CustomException(e,sys)
    
    

        
        

        
        

        
        
