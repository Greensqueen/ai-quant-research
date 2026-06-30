import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, look_back: int = 60, units: int = 50, epochs: int = 50, batch_size: int = 32):
        super().__init__()
        self.look_back = look_back
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    
    def create_dataset(self, data: np.ndarray) -> tuple:
        X, y = [], []
        for i in range(self.look_back, len(data)):
            X.append(data[i-self.look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        data = pd.concat([X, y], axis=1).values
        data_scaled = self.scaler.fit_transform(data)
        
        train_size = int(len(data_scaled) * 0.8)
        train_data = data_scaled[:train_size]
        
        X_train, y_train = self.create_dataset(train_data)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        self.model = Sequential()
        self.model.add(LSTM(units=self.units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=self.units, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping]
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        data = X.values
        data_scaled = self.scaler.transform(data)
        
        X_test, _ = self.create_dataset(data_scaled)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predictions = self.model.predict(X_test)
        
        dummy_data = np.zeros((len(predictions), data.shape[1]))
        dummy_data[:, 0] = predictions.flatten()
        predictions_original = self.scaler.inverse_transform(dummy_data)[:, 0]
        
        return predictions_original
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        predictions = self.predict(X)
        y_values = y.values[-len(predictions):]
        
        return {
            'mse': mean_squared_error(y_values, predictions),
            'rmse': np.sqrt(mean_squared_error(y_values, predictions)),
            'mae': mean_absolute_error(y_values, predictions),
            'r2': r2_score(y_values, predictions),
            'mape': np.mean(np.abs((y_values - predictions) / y_values)) * 100
        }