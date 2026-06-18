import json
import os
import shutil

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.schemas import PredictionRequest, PredictionResponse
from pipelines.train_pipeline import TrainingPipeline

app = FastAPI(title="MLOps Platform API", version="1.0.0")

# Setup templates and static files
os.makedirs("api/templates", exist_ok=True)
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

@app.get("/history")
async def get_history():
    history_path = "models_store/history.json"
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    return []

@app.get("/", response_class=HTMLResponse)
async def get_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        pipeline = TrainingPipeline(data_path=file_path, target_col=target_column)
        results = pipeline.run()
        return JSONResponse(content={
            "message": "Model trained successfully",
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema():
    """Returns expected feature names and a sample input for the current model."""
    artifacts = TrainingPipeline.load_artifacts()
    if not artifacts:
        raise HTTPException(status_code=404, detail="No trained model found.")
    
    preprocessor = artifacts['preprocessor']
    
    return {
        "feature_names": preprocessor.feature_names,
        "categorical_features": preprocessor.categorical_columns,
        "numerical_features": preprocessor.numerical_columns,
        "dropped_columns": preprocessor.dropped_columns,
        "task_type": preprocessor.task_type,
        "target_classes": getattr(preprocessor, 'target_classes', []),
        "sample_input": preprocessor.get_sample_input()
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    artifacts = TrainingPipeline.load_artifacts()
    if not artifacts:
        raise HTTPException(
            status_code=400, 
            detail="No trained model found. Please train a model first."
        )
        
    preprocessor = artifacts['preprocessor']
    model = artifacts['model']
    
    df = pd.DataFrame([request.features])
    
    # Detect which features were missing from the request
    all_features = preprocessor.feature_names
    provided_features = list(request.features.keys())
    missing_features = [f for f in all_features if f not in provided_features]
    extra_features = [f for f in provided_features if f not in all_features]

    try:
        X_processed = preprocessor.transform(df)
        raw_prediction = model.predict(X_processed)[0]
        human_label = preprocessor.decode_prediction(raw_prediction)
        
        # Confidence score
        probability = None
        if preprocessor.task_type == 'classification' and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_processed)[0]
            probability = float(max(proba))
        
        response = {
            "prediction": human_label,
            "raw_value": float(raw_prediction),
            "task_type": preprocessor.task_type,
            "confidence": probability,
            "model_used": artifacts.get('model_name', type(model).__name__),
        }
        
        # Inform user about auto-filled and ignored fields
        if missing_features:
            response["auto_filled"] = missing_features
            response["note"] = f"{len(missing_features)} missing feature(s) filled with training defaults: {missing_features}"
        if extra_features:
            response["ignored"] = extra_features

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
