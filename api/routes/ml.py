from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ml.linear_regression import LinearRegressionModel
from ml.random_forest import RandomForestModel
from ml.xgboost_model import XGBoostModel
from ml.lightgbm_model import LightGBMModel

router = APIRouter()

class MLRequest(BaseModel):
    data: list
    model_type: str = "xgboost"
    target_column: str = "Close"
    days_ahead: int = 1

class PredictionRequest(BaseModel):
    model_type: str = "xgboost"
    features: list

@router.post("/train")
def train_model(request: MLRequest):
    try:
        df = pd.DataFrame(request.data)
        
        if request.model_type == "linear_regression":
            model = LinearRegressionModel()
        elif request.model_type == "random_forest":
            model = RandomForestModel()
        elif request.model_type == "xgboost":
            model = XGBoostModel()
        elif request.model_type == "lightgbm":
            model = LightGBMModel()
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")
        
        X, y = model.prepare_features(df, target_column=request.target_column, days_ahead=request.days_ahead)
        X_train, X_test, y_train, y_test = model.split_data(X, y)
        
        model.train(X_train, y_train)
        train_metrics = model.evaluate(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)
        
        feature_importance = model.get_feature_importance(X.columns.tolist())
        
        return {
            "model_type": request.model_type,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance.to_dict(orient="records"),
            "message": "Model trained successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict")
def predict(request: PredictionRequest):
    try:
        features_df = pd.DataFrame(request.features)
        
        if request.model_type == "linear_regression":
            model = LinearRegressionModel()
        elif request.model_type == "random_forest":
            model = RandomForestModel()
        elif request.model_type == "xgboost":
            model = XGBoostModel()
        elif request.model_type == "lightgbm":
            model = LightGBMModel()
        else:
            raise ValueError(f"Unknown model type: {request.model_type}")
        
        predictions = model.predict(features_df)
        
        return {
            "model_type": request.model_type,
            "predictions": predictions.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare")
def compare_models(request: MLRequest):
    try:
        df = pd.DataFrame(request.data)
        
        models = {
            "linear_regression": LinearRegressionModel(),
            "random_forest": RandomForestModel(),
            "xgboost": XGBoostModel(),
            "lightgbm": LightGBMModel()
        }
        
        results = {}
        for name, model in models.items():
            try:
                X, y = model.prepare_features(df, target_column=request.target_column, days_ahead=request.days_ahead)
                X_train, X_test, y_train, y_test = model.split_data(X, y)
                model.train(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)
                results[name] = metrics
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))