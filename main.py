from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
import os
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR
from fastapi import FastAPI, File, UploadFile
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Request
import io
from sensor.constant import *
import numpy as np
from datetime import datetime
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from io import BytesIO
import joblib
from sensor.ml.model.estimator import SensorModel
import time

app = FastAPI()
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
        try:
            if not file.filename.endswith(".csv"):
                logging.error("Invalid file format. Expected CSV.")
                return Response(content="Invalid file format. Please upload a CSV file.", status_code=400)
            contents = file.file.read()
            buffer = BytesIO(contents)
            df = pd.read_csv(buffer)
        except Exception as e:
            logging.error(f"Failed to read CSV file: {e}")
            return Response(content="Error reading the CSV file.", status_code=500)
        

        # Read the CSV file into a DataFrame
        try:
            if df is None or df.empty:
                logging.error("Uploaded file is empty or invalid.")
                return Response(content="Uploaded file is empty or invalid.", status_code=400)
            if 'class' in df.columns:
                df.drop(columns=["class"], inplace=True) # dropping the class label
                df.replace(['na', 'NaN', 'N/A', ''], np.nan, inplace=True)
              # Drop features with max null values
                df.drop(['br_000','bq_000','bp_000','bo_000','ab_000','cr_000','bn_000','cd_000'] , inplace=True, errors='ignore')
    
        except Exception as e:
            logging.error(f"Failed to replace N/A values: {e}")
            return Response(content="Error replacing N/A values", status_code=500)

        try:
            # Drop the target column (if it exists) and apply transformation
            #input_features_df = df.drop(columns=["TARGET_COLUMN"], axis=1, errors='ignore')
            preprocessor = joblib.load(r"artifact\01_09_2025_23_44_07\data_transformation\transformed_object\preprocessing.pkl")
            expected_features = preprocessor.feature_names_in_  # For consistency

        # Align columns with preprocessor
            for col in expected_features:
                if col not in df.columns:
                    df[col] = np.nan  # Add missing columns

        # Ensure the DataFrame has the correct columns in order
            df = df[expected_features]

        # Transform the input features

            logging.info(f"Transformed input shape: {df.shape}")
        except Exception as e:
            logging.error(f"Failed to apply data transformation: {e}")
            return Response(content="Error applying data transformation.", status_code=500)
        

        # Initialize ModelResolver to load the best model
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            logging.error("Model is not available.")
            return Response(content="Model is not available", status_code=404)

        best_model_path = model_resolver.get_best_model_path()

        try:
            model = load_object(file_path=best_model_path)
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            return Response(content="Error loading the model.", status_code=500)

        # Make predictions using the model
        try:
             sensor_model = SensorModel(preprocessor=preprocessor, model=model)  # Create instance of SensorModel
             y_pred = sensor_model.predict(df)  
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return Response(content="Error during prediction.", status_code=500)

        # Update DataFrame with predictions
        df['predicted_column'] = y_pred

        # Convert DataFrame to HTML for the response
        return Response(content=df.to_csv(), media_type="text/html")
        

   
if __name__=="__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
