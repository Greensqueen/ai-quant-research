import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        predictions = self.predict(X)
        return {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(self.model.coef_)
        })
        importance = importance.sort_values('importance', ascending=False)
        return importance