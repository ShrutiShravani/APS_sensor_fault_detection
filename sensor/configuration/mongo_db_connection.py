import pymongo
from sensor.constant.database import DATABASE_NAME
import certifi
ca=certifi.where()
from sensor.constant.env_variable import MONGODB_URL_KEY
import os,sys

class MongoDBClient:
    client= None

    def __init__(self,database_name= DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url=os.getenv(MONGODB_URL_KEY)
                if not mongo_db_url:
                     raise ValueError("MongoDB URL is not set in the environment variable.")
                else:
                    MongoDBClient.client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
                self.client= MongoDBClient.client
                self.database=self.client[database_name]
                self.database_name= database_name
        except Exception as e:
            raise e
