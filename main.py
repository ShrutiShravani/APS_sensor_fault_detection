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
        # Check if the uploaded file is a CSV
        if not file.filename.endswith(".csv"):
            logging.error("Invalid file format. Expected CSV.")
            return Response(content="Invalid file format. Please upload a CSV file.", status_code=400)

        # Convert file to binary data
        binary_file = await file.read()

        # Use io.BytesIO to convert binary data to a file-like object
        file_like_object = io.BytesIO(binary_file)
        
        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(file_like_object)
        except Exception as e:
            logging.error(f"Failed to read CSV file: {e}")
            return Response(content="Error reading the CSV file.", status_code=500)

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
            y_pred = model.predict(df)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return Response(content="Error during prediction.", status_code=500)

        # Update DataFrame with predictions
        df['predicted_column'] = y_pred

        # Reverse mapping of target values if necessary
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)

        # Convert DataFrame to HTML for the response
        return Response(content=df.to_html(), media_type="text/html")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return Response(content=f"Unexpected error occurred: {str(e)}", status_code=500)

if __name__=="__main__":
    #main()
    # set_env_variable(env_file_path)
    app_run(app, host=APP_HOST, port=APP_PORT)
