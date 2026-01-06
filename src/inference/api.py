"""
FastAPI service for real-time churn prediction.

Usage:
    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import logging

from ..models.registry import ModelRegistry
from ..feature_engineering import FeatureStore
from ..utils import load_config, setup_logging

logger = setup_logging(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn prediction service",
    version="1.0.0"
)

# Global model cache
_model = None
_model_metadata = None


class PredictionRequest(BaseModel):
    """Request body for churn prediction."""
    customer_id: str


class PredictionResponse(BaseModel):
    """Response body for churn prediction."""
    customer_id: str
    churn_probability: float
    risk_segment: str
    model_version: str


def load_production_model():
    """Load the production model (cached)."""
    global _model, _model_metadata
    
    if _model is None:
        registry = ModelRegistry()
        model_entry = registry.get_production_model()
        
        if not model_entry:
            raise ValueError("No production model found")
        
        _model = joblib.load(model_entry['model_path'])
        _model_metadata = model_entry
        logger.info(f"Loaded production model: {model_entry['model_version']}")
    
    return _model, _model_metadata


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_production_model()
    logger.info("API server started")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "churn-prediction-api"}


@app.get("/model-info")
async def model_info():
    """Get information about the current model."""
    _, metadata = load_production_model()
    return {
        "model_version": metadata['model_version'],
        "metrics": metadata['metadata']['metrics'],
        "registered_at": metadata['registered_at'],
    }


@app.post("/predict_churn", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """
    Predict churn probability for a customer.
    
    Args:
        request: Prediction request with customer_id
        
    Returns:
        Prediction response with churn probability and risk segment
    """
    try:
        # Load model
        model, metadata = load_production_model()
        
        # Get features from online store
        config = load_config('configs/features.yaml')
        feature_store = FeatureStore(config)
        
        features = feature_store.get_online_features(request.customer_id)
        
        if features is None:
            raise HTTPException(
                status_code=404,
                detail=f"Customer {request.customer_id} not found in feature store"
            )
        
        # Prepare features for prediction
        df = pd.DataFrame([features])
        exclude_cols = ['customer_id', 'snapshot_date', 'churn_label_next_period', 'months_until_churn', 'year_month', 'tenure_bucket']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = pd.Categorical(X[col]).codes
        X = X.fillna(0)
        
        # Predict
        churn_prob = float(model.predict_proba(X)[0, 1])
        
        # Determine risk segment
        if churn_prob >= 0.6:
            risk_segment = "high"
        elif churn_prob >= 0.4:
            risk_segment = "medium"
        else:
            risk_segment = "low"
        
        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=churn_prob,
            risk_segment=risk_segment,
            model_version=metadata['model_version']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server():
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_server()
