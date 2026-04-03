import os
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from pipelines.train_pipeline import TrainingPipeline
from api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="MLOps Platform API", version="1.0.0")

# Setup templates
os.makedirs("api/templates", exist_ok=True)
templates = Jinja2Templates(directory="api/templates")

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

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    artifacts = TrainingPipeline.load_artifacts()
    if not artifacts:
        raise HTTPException(status_code=400, detail="No trained model found. Please train a model first.")
        
    preprocessor = artifacts['preprocessor']
    model = artifacts['model']
    
    # Convert input to DataFrame
    df = pd.DataFrame([request.features])
    
    try:
        X_processed = preprocessor.transform(df)
        prediction = model.predict(X_processed)
        
        # If classification, inverse transform prediction if needed
        if preprocessor.task_type == 'classification':
            prediction = preprocessor.label_encoder.inverse_transform(prediction)
            
        return PredictionResponse(prediction=prediction.tolist()[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
