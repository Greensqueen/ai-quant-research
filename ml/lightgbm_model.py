import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .base_model import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, random_state: int = 42):
        super().__init__()
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
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
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        return importance