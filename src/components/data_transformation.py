import os
import sys

from src.logging import logging
from src.exception import CustomException
from src.utils import save_object

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
    def __init__(self):
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
    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preprocessor_obj=self.get_data_transformation()
            
            logging.info("reading the training and testing data")
            logging.info(f"Training DataFrame : \n{train_df.head(5).to_string()}")
            logging.info(f"Testing DataFrame : \n{test_df.head(5).to_string()}")
            
            input_features_train_df=train_df.drop('accident_risk', axis=1)
            target_features_train_df=train_df['accident_risk']
            
            input_features_test_df=test_df.drop('accident_risk', axis=1)
            target_features_test_df=test_df['accident_risk']
            
            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df)
            
            train_arr=np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr=np.c_[input_features_test_arr, np.array(target_features_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preperocessor_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Preprocessing Pickle file saved!!!")
            
            return (
                train_arr, test_arr
            )
            
        except Exception as e:
            logging.info("error occured during the data transformation")
            raise CustomException(e,sys)
            
    
    

        
        

        
        

        
        
