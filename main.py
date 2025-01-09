from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import  SensorException
from sensor.logger import logging
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.constant.application import APP_HOST,APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from fastapi import FastAPI, File, UploadFile,Request
import os,sys
import pandas as pd
from sensor.constant.env_variable import MONGODB_URL_KEY
from pymongo import MongoClient



if __name__=="__main__":
      train_pipeline = TrainPipeline()
      train_pipeline.run_pipeline()