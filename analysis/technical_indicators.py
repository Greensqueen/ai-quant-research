import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self):
        pass
    
    def add_sma(self, df: pd.DataFrame, window: int = 20, column: str = 'Close') -> pd.DataFrame:
        df[f'SMA_{window}'] = df[column].rolling(window=window).mean()
        return df
    
    def add_ema(self, df: pd.DataFrame, window: int = 12, column: str = 'Close') -> pd.DataFrame:
        df[f'EMA_{window}'] = df[column].ewm(span=window, adjust=False).mean()
        return df
    
    def add_rsi(self, df: pd.DataFrame, window: int = 14, column: str = 'Close') -> pd.DataFrame:
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    def add_macd(self, df: pd.DataFrame, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
        df['MACD'] = df['Close'].ewm(span=fast_window, adjust=False).mean() - df['Close'].ewm(span=slow_window, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, column: str = 'Close') -> pd.DataFrame:
        df['BB_Mid'] = df[column].rolling(window=window).mean()
        df['BB_Upper'] = df['BB_Mid'] + 2 * df[column].rolling(window=window).std()
        df['BB_Lower'] = df['BB_Mid'] - 2 * df[column].rolling(window=window).std()
        return df
    
    def add_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=window).mean()
        return df
    
    def add_obv(self, df: pd.DataFrame, column: str = 'Close', volume_column: str = 'Volume') -> pd.DataFrame:
        df['OBV'] = (np.sign(df[column].diff()) * df[volume_column]).cumsum()
        return df
    
    def add_momentum(self, df: pd.DataFrame, window: int = 10, column: str = 'Close') -> pd.DataFrame:
        df['Momentum'] = df[column] - df[column].shift(window)
        return df
    
    def add_stochastic(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['%D'] = df['%K'].rolling(window=3).mean()
        return df
    
    def add_adx(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        up_move = df['High'] - df['High'].shift()
        down_move = df['Low'].shift() - df['Low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr_sma = tr.rolling(window=window).sum()
        plus_di = 100 * (plus_dm.rolling(window=window).sum() / tr_sma)
        minus_di = 100 * (minus_dm.rolling(window=window).sum() / tr_sma)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=window).mean()
        return df
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_sma(df, window=20)
        df = self.add_sma(df, window=50)
        df = self.add_ema(df, window=12)
        df = self.add_ema(df, window=26)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_obv(df)
        df = self.add_momentum(df)
        df = self.add_stochastic(df)
        df = self.add_adx(df)
        return df