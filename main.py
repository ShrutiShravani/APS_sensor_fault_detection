from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
import os
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
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
from sensor.entity.config_entity import DataTransformationConfig,TrainingPipelineConfig
from sensor.constant import *
import numpy as np
from datetime import datetime
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from io import BytesIO
import joblib

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

def get_data_transformer_object() -> Pipeline:
    try:
        robust_scaler = RobustScaler()
        simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
        preprocessor = Pipeline(
            steps=[
                ("Imputer", simple_imputer),  # Replace missing values with zero
                ("RobustScaler", robust_scaler)  # Scale features and handle outliers
            ]
        )
        return preprocessor
    except Exception as e:
        raise Exception(f"Error in data transformation: {e}") from e


@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Check if the uploaded file is a CSV
        if not file.filename.endswith(".csv"):
            logging.error("Invalid file format. Expected CSV.")
            return Response(content="Invalid file format. Please upload a CSV file.", status_code=400)

        # Convert file to binary data
        binary_file = await file.read()

        # Use io.BytesIO to convert binary data to a file-like object
        file_like_object = BytesIO(binary_file)

         # Check if file is empty
        if file_like_object.getbuffer().nbytes == 0:
            logging.error("Uploaded file is empty.")
            return Response(content="Uploaded file is empty.", status_code=400)
        

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(file_like_object)
            if df is None or df.empty:
                logging.error("Uploaded file is empty or invalid.")
                return Response(content="Uploaded file is empty or invalid.", status_code=400)
            df=df.replace(['na', 'NaN', 'N/A', ''], np.nan, inplace=True)
    
        except Exception as e:
            logging.error(f"Failed to read CSV file: {e}")
            return Response(content="Error reading the CSV file.", status_code=500)

        try:
            # Drop the target column (if it exists) and apply transformation
            #input_features_df = df.drop(columns=["TARGET_COLUMN"], axis=1, errors='ignore')
            preprocessor = get_data_transformer_object()
            if preprocessor is None:
                 logging.error("Failed to initialize data transformer object.")
                 return Response(content="Error initializing data transformer.", status_code=500)
            preprocessor_object = preprocessor.fit(df)
            transformed_input_features = preprocessor_object.transform(df)
            logging.info(f"Transformed input shape: {transformed_input_features.shape}")
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
            y_pred = model.predict(transformed_input_features)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return Response(content="Error during prediction.", status_code=500)

        # Update DataFrame with predictions
        df['predicted_column'] = y_pred

        # Convert DataFrame to HTML for the response
        return Response(content=df.to_html(), media_type="text/html")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return Response(content=f"Unexpected error occurred: {str(e)}", status_code=500)

if __name__=="__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
