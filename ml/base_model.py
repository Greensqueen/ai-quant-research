import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.scaler = None
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        pass
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'Close', days_ahead: int = 1) -> tuple:
        df = df.copy()
        df['Target'] = df[target_column].shift(-days_ahead)
        df = df.dropna()
        
        X = df.drop(['Target', target_column], axis=1)
        y = df['Target']
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, train_size: float = 0.8) -> tuple:
        split_idx = int(len(X) * train_size)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test