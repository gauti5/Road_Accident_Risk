import os, sys

from src.logging import logging
from src.exception import CustomException

from pathlib import Path
from src.utils import load_object

import pandas as pd


class predict_pipeline:
     
    def __init__(self):
        pass
    def predict(self, features):
        try:
            
            model_path=os.path.join("Artifacts", "Model.pkl")
            preprocessor_path=os.path.join("Artifacts", "preprocessor.pkl")
                
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
                
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Error occured during the prediction")
            raise CustomException(e,sys)


        
class CustomData:
    def __init__(self,
                 num_lanes:int,
                 curvature:float,
                 speed_limit:int,
                 lighting:object,
                 weather:object,
                 road_signs_present:bool,
                 public_road:bool,
                 time_of_day:object,
                 holiday:bool,
                 school_season:bool,
                 num_reported_accidents:int):
        self.num_lanes=num_lanes,
        self.curvature=curvature,
        self.speed_limit=speed_limit,
        self.lighting=lighting,
        self.weather=weather,
        self.road_signs_present=road_signs_present,
        self.public_road=public_road,
        self.time_of_day=time_of_day,
        self.holiday=holiday,
        self.school_season=school_season,
        self.num_reported_accidents=num_reported_accidents
    
    def get_data_as_a_frame(self):
        try:
            custom_data_input_dict={
                'num_lanes':[self.num_lanes],
                'curvature':[self.curvature],
                'speed_limit':[self.speed_limit],
                'lighting':[self.lighting],
                'weather':[self.weather],
                'road_signs_present':[self.road_signs_present],
                'public_road':[self.public_road],
                'time_of_day':[self.time_of_day],
                'holiday':[self.holiday],
                'school_season':[self.school_season],
                'num_reported_accidents':[self.num_reported_accidents]
                
                
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered!!")
            return df
        
        except Exception as e:
            raise CustomException(e,sys)
            
        