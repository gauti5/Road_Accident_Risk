import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL=os.getenv("MONGO_DB_URL")

# print(MONGODB_URL)

import certifi
ca=certifi.where()

# print(ca)

from src.exception import CustomException
from src.logging import logging
import pandas as pd
import numpy as np
import pymongo
from pymongo.mongo_client import MongoClient

class DataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
        
    def csv_to_json_conversion(self, file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
        
    def insert_data_to_Mongo(self, records, database, collection):
        try:
            
            self.records=records
            self.datbase=database
            self.collection=collection
            self.mongo_client=MongoClient(MONGODB_URL)
            self.database=self.mongo_client[self.database]
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=='__main__':
    file_path='Notebook\Data.csv'
    database='SandeepAI'
    collection='RoadData'
    obj=DataExtract()
    records=obj.csv_to_json_conversion(file_path=file_path)
    print(records)
    no_of_records=obj.insert_data_to_Mongo(records, database, collection)
    print(no_of_records)
    
    
    
