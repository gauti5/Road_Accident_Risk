import os, sys

from src.logging import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("Artifacts", "Raw_Data.csv")
    train_data_path:str=os.path.join("Artifacts", "Train_Data.csv")
    test_data_path:str=os.path.join("Artifacts", "Test_Data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started!!!")
        try:
            Data=pd.read_csv("Notebook/Data.csv")
            print(Data)
            logging.info(Data.head(10))
            logging.info(Data.shape)
            logging.info(Data.describe)
            logging.info(Data.info())
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            Data.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)
            
            logging.info("Splitting the Data into train and test")
            
            train_data, test_data=train_test_split(Data, test_size=0.3, random_state=24)
            
            logging.info("Splitting Completed!!!")
            
            logging.info("training data information - ")
            logging.info(train_data.shape)
            logging.info(train_data.head(5))
            logging.info("Testing Data Information - ")
            logging.info(test_data.shape)
            logging.info(test_data.head(5))
            
            train_data.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, header=True, index=False)
            
            logging.info("Data Ingestion Completed!!!")
            
            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Error occured during the data ingestion!!")
            raise CustomException(e,sys)
        
        
if __name__=='__main__':
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    